function out = Jp(mu,Z)
%JP Summary of this function goes here
%   Detailed explanation goes here
out = besselj(mu-1,Z)-(mu/Z)*besselj(mu,Z);
end


