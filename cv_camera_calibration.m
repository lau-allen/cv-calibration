%===================================================
% Computer Vision: Camera Calibration
% Allen Lau 
%===================================================

% A. Calibration Pattern Design %
% Generate data of a “virtual” 3D cube

% Pw (calibration pattern) holds 32 points on two surfaces 
% (Xw = 1 and Yw = 1) of a cube. Values are measured in meters.
% There are 4x4 uniformly distributed points on each surface.

% define variable to define which row is being populated
cnt = 1;

% plane : Xw = 1
for i=0.2:0.2:0.8,
 for j=0.2:0.2:0.8,
   Pw(cnt,:) = [1 i j];
   cnt = cnt + 1;
 end
end

% plane : Yw = 1
for i=0.2:0.2:0.8,
 for j=0.2:0.2:0.8,
   Pw(cnt,:) = [i 1 j];
   cnt = cnt + 1;
 end
end

% plot calibration cube
figure; 
plot3(Pw(:,1), Pw(:,2), Pw(:,3), '+');
% define axes bounds
axis([0 1 0 1 0 1]);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
grid on;
title('Figure 1: Calibration Cube');

%%
% B. Virtual Camera & Images %

% Design a “virtual” camera with known intrinsic parameters 
% including focal length f, image center (ox, oy) and pixel size (sx, sy).

% Defining camera instrinsic parameters 
% focal length
f = 0.016; % m
% image frame size 
img_frame_size = [512, 512]; % pixels
% image center
Ox = 256; % pixels
Oy = 256; % pixels
% image sensor = 8.8 mm * 6.6 mm
% pixel size 
Sx = 0.0088/512.0;
Sy = 0.0066/512.0;
% effective focal length 
Fx = f/Sx;
Fy = f/Sy;
% aspect ratio
asr = Fx/Fy;

% Extrinsic parameters : R = RaRbRr

% initial angle for z-axis rotation
% define gamma as 30 degrees 
gamma = 30.0*pi/180.0;
% rotation matrix about the z-axis
Rr = [ [cos(gamma) -sin(gamma) 0];
       [sin(gamma) cos(gamma)  0];
       [  0          0         1]; ];
% initial angle for y-axis rotation matrix 
beta = 0.0*pi/180.0;
% rotation matrix about the y-axis
Rb = [ [cos(beta) 0 -sin(beta)];
       [0         1       0];
       [sin(beta) 0  cos(beta)]; ];
% initial angle for x-axis rotation matrix 
alpha = -120.0*pi/180.0;
%alpha = 0.0*pi/180.0;
% rotation about the x-axis 
Ra = [ [1      0                0];
       [0   cos(alpha)  -sin(alpha)];
       [0   sin(alpha)   cos(alpha)]; ];

% define rotation matrix
R = Ra*Rb*Rr;

% define translation matrix
T = [0 0 4]';

% Generate Image coordinates
% surface Xw = 1
cnt = 1;
for cnt = 1:1:16,
   % world to camera 
   Pc(cnt,:) = (R*Pw(cnt,:)' + T)';
   % camera to image using projective equations; subtract from image center
   % to get it into image frame 
   p(cnt,:)  = [(Ox - Fx*Pc(cnt,1)/Pc(cnt,3)) (Oy - Fy*Pc(cnt,2)/Pc(cnt,3))];
end
figure; 
plot(p(:,1), p(:,2), 'b+');
axis([0 512 0 512]);
grid on;
hold on;

% surface Yw = 1
for cnt = 17:1:32,
   % world to camera 
   Pc(cnt,:) = (R*Pw(cnt,:)' + T)';
   % camera to image using projective equations; subtract from image center
   % to get it into image frame 
   p(cnt,:)  = [(Ox - Fx*Pc(cnt,1)/Pc(cnt,3)) (Oy - Fy*Pc(cnt,2)/Pc(cnt,3))];
end
plot(p(17:32,1), p(17:32,2), 'b+');
title('Figure 2: Calibration Image, gamma = 30 deg')
grid on;
hold off; 

% % re-defining the origin of the p array to top left by convention 
% for i = 1:length(p)
%     data = p(i,:);
%     p(i,:) = [data(1),512-data(2)];
% end

%%
% C. Direction Calibration Method %
% Estimate the intrinsic (fx, fy, aspect ratio a, image center (ox,oy)) 
% and extrinsic (R, T and further alpha, beta, gamma) parameters. 
% Use SVD to solve the homogeneous linear system and the least square 
% problem, and to enforce the orthogonality constraint on the estimate of R. 


% i. Use the accurately simulated data (both 3D world coordinates and 
% 2D image coordinates) to the algorithms, and compare the results 
% with the "ground truth” data (which are given in step (a) and step (b)).
% In the direct calibration method, can use the knowledge of the image 
% center (in the homogeneous system to find extrinsic parameters) and 
% the aspect ratio (in the Orthocenter theorem method to find image center). 

% given information: p = matrix of (x_im, y_im) coordinates of the
% calibration points, Pw = matrix of (X_w, Y_w) coordinates 

% define empty matrix for A 
A = zeros(length(Pw),8);

% populate A 
for i = 1:length(Pw)
    data_pw = Pw(i,:);
    data_p = [p(i,1)-256,p(i,2)-256]; %subtracting 256 to get point into x', y' frame 
    % build each row for each world - image point pair 
    row = [data_p(1)*data_pw(1),data_p(1)*data_pw(2),data_p(1)*data_pw(3),data_p(1),-data_p(2)*data_pw(1),-data_p(2)*data_pw(2),-data_p(2)*data_pw(3),-data_p(2)];
    A(i,:) = row; 
end

% compute SVD of A 
[U,D,V_transposed] = svd(A);

% compute v_bar = ce_8 = 8th row corresponding to the only zero singlular
% value lamba_eight = 0 of A^T*A 
% extract v_bar from V_transposed
v_bar = V_transposed(8,:);

% gamma magnitude
gamma_mag = sqrt(v_bar(1)^2 + v_bar(2)^2 + v_bar(3)^2);
%disp(['Magnitude of Gamma: ', num2str(gamma_mag)])

% checking that r21^2 + r_22^2 + r_23^2 = 1
check = (v_bar(1)/gamma_mag)^2 + (v_bar(2)/gamma_mag)^2 + (v_bar(3)/gamma_mag)^2;
disp(['r21^2 + r_22^2 + r_23^2 = 1?: ',num2str(check)])

% alpha 
asr_est = sqrt(v_bar(5)^2+v_bar(6)^2+v_bar(7)^2)/(gamma_mag);
%disp(['asr: ',num2str(asr_est)])
% checking that r11^2 + r_12^2 + r_13^2 = 1
check2 = (v_bar(5)/(asr_est*gamma_mag))^2 + (v_bar(6)/(asr_est*gamma_mag))^2 + (v_bar(7)/(asr_est*gamma_mag))^2;
disp(['r11^2 + r_12^2 + r_13^2 = 1?:',num2str(check2)])

% constructing first two components of T 
Ty_est = v_bar(4)/gamma_mag;
Tx_est = v_bar(8)/(asr_est*gamma_mag);

% constructing first two rows of R 
R1 = [v_bar(5)/(asr_est*gamma_mag),v_bar(6)/(asr_est*gamma_mag),v_bar(7)/(asr_est*gamma_mag)];
R2 = [v_bar(1)/gamma_mag,v_bar(2)/gamma_mag,v_bar(3)/gamma_mag];

% computing sign, s
ex_world_point = Pw(1,:);
ex_img_point = p(1,:);
X_sign = R1(1)*ex_world_point(1) + R1(2)*ex_world_point(2) + R1(3)*ex_world_point(3) + Tx_est;

% assumption that s is positive; X and x should have opposite signs
% if X_sign is positive, then assumption of s = 1 is true
% we know that x is positive by inspecting the point directly
if (X_sign > 0)
    s = 1;
else
    s = -1;
end

% applying the sign
R1 = s*R1; 
R2 = s*R2;
Tx_est = s*Tx_est;
Ty_est = s*Ty_est; 

%disp(['Tx: ',num2str(Tx_est)])
%disp(['Ty: ',num2str(Ty_est)])

% computing row 3 
R3 = cross(R1',R2')';

% constructing rotation matrix
R_hat = [R1;R2;R3];
% enforce orthogonality 
[U_R_hat,D_R_hat,V_transposed_R_hat] = svd(R_hat);
% compute R 
R_est = U_R_hat*eye(3,3)*V_transposed_R_hat; 
%disp('R: ')
%disp(R_est)

%computing Tz, Fx, Fy
%populating matrices 
a = zeros(length(p),2);
b = zeros(length(p),1);
for i = 1:length(p)
    data_p = p(i,:);
    data_pw = Pw(i,:);
    a(i,:) = [data_p(1),R_est(1,1)*data_pw(1)+R_est(1,2)*data_pw(2)+R_est(1,3)*data_pw(3)+Tx_est];
    b(i,1) = R_est(3,1)*data_pw(1)+R_est(3,2)*data_pw(2)+R_est(3,3)*data_pw(3);
end
%computing Tz and fx 
A_inv = pinv(a'*a);
values = A_inv*a'*b;
Tz_est = values(1);
fx_est = values(2);

% disp(['Tz: ',num2str(Tz_est)]);
% disp(['fx: ',num2str(fx_est)]);

%compute fy
fy_est = fx_est / asr_est;
%disp(['fy: ',num2str(fy_est)]);

% comparing ground truths to estimated values 
disp('asr')
disp([asr,asr_est])
disp('Tx')
disp([T(1),Tx_est])
disp('Ty')
disp([T(2),Ty_est])
disp('Tz')
disp([T(3),Tz_est])
disp('R: ')
disp(R)
disp(R_est)
disp('fx')
disp([Fx,fx_est])
disp('fy')
disp([Fy,fy_est])

%%
% how the initial estimation of image center affects the estimating 
% of the remaining parameters (5 points), by experimental results

% given information: p = matrix of (x_im, y_im) coordinates of the
% calibration points, Pw = matrix of (X_w, Y_w) coordinates 

% define empty matrix for A 
A = zeros(length(Pw),8);

%defining image center 
ox = 128;
oy = 128; 

% populate A 
for i = 1:length(Pw)
    data_pw = Pw(i,:);
    data_p = [p(i,1)-ox,p(i,2)-oy]; %subtracting 256 to get point into x', y' frame 
    % build each row for each world - image point pair 
    row = [data_p(1)*data_pw(1),data_p(1)*data_pw(2),data_p(1)*data_pw(3),data_p(1),-data_p(2)*data_pw(1),-data_p(2)*data_pw(2),-data_p(2)*data_pw(3),-data_p(2)];
    A(i,:) = row; 
end

% compute SVD of A 
[U,D,V_transposed] = svd(A);

% compute v_bar = ce_8 = 8th row corresponding to the only zero singlular
% value lamba_eight = 0 of A^T*A 
% extract v_bar from V_transposed
v_bar = V_transposed(8,:);

% gamma magnitude
gamma_mag = sqrt(v_bar(1)^2 + v_bar(2)^2 + v_bar(3)^2);
%disp(['Magnitude of Gamma: ', num2str(gamma_mag)])

% checking that r21^2 + r_22^2 + r_23^2 = 1
check = (v_bar(1)/gamma_mag)^2 + (v_bar(2)/gamma_mag)^2 + (v_bar(3)/gamma_mag)^2;
disp(['r21^2 + r_22^2 + r_23^2 = 1?: ',num2str(check)])

% alpha 
asr_est = sqrt(v_bar(5)^2+v_bar(6)^2+v_bar(7)^2)/(gamma_mag);
%disp(['asr: ',num2str(asr_est)])
% checking that r11^2 + r_12^2 + r_13^2 = 1
check2 = (v_bar(5)/(asr_est*gamma_mag))^2 + (v_bar(6)/(asr_est*gamma_mag))^2 + (v_bar(7)/(asr_est*gamma_mag))^2;
disp(['r11^2 + r_12^2 + r_13^2 = 1?:',num2str(check2)])

% constructing first two components of T 
Ty_est = v_bar(4)/gamma_mag;
Tx_est = v_bar(8)/(asr_est*gamma_mag);

% constructing first two rows of R 
R1 = [v_bar(5)/(asr_est*gamma_mag),v_bar(6)/(asr_est*gamma_mag),v_bar(7)/(asr_est*gamma_mag)];
R2 = [v_bar(1)/gamma_mag,v_bar(2)/gamma_mag,v_bar(3)/gamma_mag];

% computing sign, s
ex_world_point = Pw(1,:);
ex_img_point = p(1,:);
X_sign = R1(1)*ex_world_point(1) + R1(2)*ex_world_point(2) + R1(3)*ex_world_point(3) + Tx_est;

% assumption that s is positive; X and x should have opposite signs
% if X_sign is positive, then assumption of s = 1 is true
% we know that x is positive by inspecting the point directly
if (X_sign > 0)
    s = 1;
else
    s = -1;
end

% applying the sign
R1 = s*R1; 
R2 = s*R2;
Tx_est = s*Tx_est;
Ty_est = s*Ty_est; 

%disp(['Tx: ',num2str(Tx_est)])
%disp(['Ty: ',num2str(Ty_est)])

% computing row 3 
R3 = cross(R1',R2')';

% constructing rotation matrix
R_hat = [R1;R2;R3];
% enforce orthogonality 
[U_R_hat,D_R_hat,V_transposed_R_hat] = svd(R_hat);
% compute R 
R_est = U_R_hat*eye(3,3)*V_transposed_R_hat; 
%disp('R: ')
%disp(R_est)

%computing Tz, Fx, Fy
%populating matrices 
a = zeros(length(p),2);
b = zeros(length(p),1);
for i = 1:length(p)
    data_p = p(i,:);
    data_pw = Pw(i,:);
    a(i,:) = [data_p(1),R_est(1,1)*data_pw(1)+R_est(1,2)*data_pw(2)+R_est(1,3)*data_pw(3)+Tx_est];
    b(i,1) = R_est(3,1)*data_pw(1)+R_est(3,2)*data_pw(2)+R_est(3,3)*data_pw(3);
end
%computing Tz and fx 
A_inv = pinv(a'*a);
values = A_inv*a'*b;
Tz_est = values(1);
fx_est = values(2);

% disp(['Tz: ',num2str(Tz_est)]);
% disp(['fx: ',num2str(fx_est)]);

%compute fy
fy_est = fx_est / asr_est;
%disp(['fy: ',num2str(fy_est)]);

% comparing ground truths to estimated values 
disp('asr')
disp([asr,asr_est])
disp('Tx')
disp([T(1),Tx_est])
disp('Ty')
disp([T(2),Ty_est])
disp('Tz')
disp([T(3),Tz_est])
disp('R: ')
disp(R)
disp(R_est)
disp('fx')
disp([Fx,fx_est])
disp('fy')
disp([Fy,fy_est])
