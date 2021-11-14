function out = Hp(mu,Z)
%HP Summary of this function goes here
%   Detailed explanation goes here
out = besselh(mu-1,2,Z)-(mu/Z)*besselh(mu,2,Z);
end

