function [U,S,V] = randpca_AB(A,B,k,its,l)
% A*B is the matrix to factorize, where A and B are sparse.
% raw is fixed to true.

% Retrieve the dimensions of A*B.
m = size(A,1);
n = size(B,2);

if(m >= n)
  
  %
  %   Apply A to a random matrix, obtaining Q.
  %
  if(isreal(A))
    Q = A* (B*(2*rand(n,l)-ones(n,l)));
  end
  
  %
  %   Form a matrix Q whose columns constitute a well-conditioned basis
  %   for the columns of the earlier Q.
  %
  if(its == 0)
    [Q,R,E] = qr(Q,0);
  end
  
  if(its > 0)
    [Q,R] = lu(Q);
  end
  
  %
  %   Conduct normalized power iterations.
  %
  for it = 1:its
    
    Q = ((Q'*A) * B)';
    
    [Q,R] = lu(Q);
    
    Q = A * (B*Q);
    
    if(it < its)
      [Q,R] = lu(Q);
    end
    
    if(it == its)
      [Q,R,E] = qr(Q,0);
    end
    
  end
  
  clear E;
  
  %
  %   SVD Q'*A to obtain approximations to the singular values
  %   and right singular vectors of A; adjust the left singular vectors
  %   of Q'*A to approximate the left singular vectors of A.
  %
  [R,S,V] = svd( (Q'*A) * B,'econ');
  U = Q*R;
  
  clear Q R;
  
  %
  %   Retain only the leftmost k columns of U,
  %   the leftmost k columns of V, and the
  %   uppermost leftmost k x k block of S.
  %
  U = U(:,1:k);
  V = V(:,1:k);
  S = S(1:k,1:k);
  
end


if(m < n)
  
  %
  %   Apply A' to a random matrix, obtaining Q.
  %
  if(isreal(A))
    Q = ( ((2*rand(l,m)-ones(l,m))*A) * B)';
  end
  
  %
  %   Form a matrix Q whose columns constitute a well-conditioned basis
  %   for the columns of the earlier Q.
  %
  if(its == 0)
    [Q,R,E] = qr(Q,0);
  end
  
  if(its > 0)
    [Q,R] = lu(Q);
  end
  
  %
  %   Conduct normalized power iterations.
  %
  for it = 1:its
    
    Q = A * (B*Q);
    
    [Q,R] = lu(Q);
    
    Q = ( (Q'*A) * B)';
    
    if(it < its)
      [Q,R] = lu(Q);
    end
    
    if(it == its)
      [Q,R,E] = qr(Q,0);
    end
    
  end
  
  clear E;
  
  %
  %   SVD A*Q to obtain approximations to the singular values
  %   and left singular vectors of A; adjust the right singular vectors
  %   of A*Q to approximate the right singular vectors of A.
  %
  [U,S,R] = svd(A*(B*Q),'econ');
  V = Q*R;
  
  clear Q R;
  
  %
  %   Retain only the leftmost k columns of U,
  %   the leftmost k columns of V, and the
  %   uppermost leftmost k x k block of S.
  %
  U = U(:,1:k);
  V = V(:,1:k);
  S = S(1:k,1:k);
  
end
