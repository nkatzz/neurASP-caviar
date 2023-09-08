
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbpugt_2,137,923). coords(lbpugt_2,211,108,923).  visible(lbpugt_2,923). happensAt(walking(lbpugt_2),923). orientation(lbpugt_4,135,923). coords(lbpugt_4,219,34,923).  visible(lbpugt_4,923). happensAt(inactive(lbpugt_4),923).
orientation(lbpugt_2,135,924). coords(lbpugt_2,209,107,924).  visible(lbpugt_2,924). happensAt(walking(lbpugt_2),924). orientation(lbpugt_4,135,924). coords(lbpugt_4,219,34,924).  visible(lbpugt_4,924). happensAt(inactive(lbpugt_4),924).
orientation(lbpugt_2,135,925). coords(lbpugt_2,207,106,925).  visible(lbpugt_2,925). happensAt(walking(lbpugt_2),925). orientation(lbpugt_4,135,925). coords(lbpugt_4,219,34,925).  visible(lbpugt_4,925). happensAt(inactive(lbpugt_4),925).
orientation(lbpugt_2,131,926). coords(lbpugt_2,205,105,926).  visible(lbpugt_2,926). happensAt(walking(lbpugt_2),926). orientation(lbpugt_4,135,926). coords(lbpugt_4,219,34,926).  visible(lbpugt_4,926). happensAt(inactive(lbpugt_4),926).
orientation(lbpugt_2,131,927). coords(lbpugt_2,205,105,927).  visible(lbpugt_2,927). happensAt(walking(lbpugt_2),927). orientation(lbpugt_4,135,927). coords(lbpugt_4,219,34,927).  visible(lbpugt_4,927). happensAt(inactive(lbpugt_4),927).
orientation(lbpugt_2,131,928). coords(lbpugt_2,204,104,928).  visible(lbpugt_2,928). happensAt(walking(lbpugt_2),928). orientation(lbpugt_4,135,928). coords(lbpugt_4,219,34,928).  visible(lbpugt_4,928). happensAt(inactive(lbpugt_4),928).
orientation(lbpugt_2,131,929). coords(lbpugt_2,204,104,929).  visible(lbpugt_2,929). happensAt(walking(lbpugt_2),929). orientation(lbpugt_4,135,929). coords(lbpugt_4,219,34,929).  visible(lbpugt_4,929). happensAt(inactive(lbpugt_4),929).
orientation(lbpugt_2,131,930). coords(lbpugt_2,204,103,930).  visible(lbpugt_2,930). happensAt(walking(lbpugt_2),930). orientation(lbpugt_4,135,930). coords(lbpugt_4,219,34,930).  visible(lbpugt_4,930). happensAt(inactive(lbpugt_4),930).
orientation(lbpugt_2,131,931). coords(lbpugt_2,203,103,931).  visible(lbpugt_2,931). happensAt(walking(lbpugt_2),931). orientation(lbpugt_4,135,931). coords(lbpugt_4,219,34,931).  visible(lbpugt_4,931). happensAt(inactive(lbpugt_4),931).
orientation(lbpugt_2,125,932). coords(lbpugt_2,203,102,932).  visible(lbpugt_2,932). happensAt(walking(lbpugt_2),932). orientation(lbpugt_4,135,932). coords(lbpugt_4,219,34,932).  visible(lbpugt_4,932). happensAt(inactive(lbpugt_4),932).
orientation(lbpugt_2,125,933). coords(lbpugt_2,203,101,933).  visible(lbpugt_2,933). happensAt(walking(lbpugt_2),933). orientation(lbpugt_4,135,933). coords(lbpugt_4,219,34,933).  visible(lbpugt_4,933). happensAt(inactive(lbpugt_4),933).
orientation(lbpugt_2,124,934). coords(lbpugt_2,202,99,934).  visible(lbpugt_2,934). happensAt(walking(lbpugt_2),934). orientation(lbpugt_4,135,934). coords(lbpugt_4,219,34,934).  visible(lbpugt_4,934). happensAt(inactive(lbpugt_4),934).
orientation(lbpugt_2,124,935). coords(lbpugt_2,201,97,935).  visible(lbpugt_2,935). happensAt(walking(lbpugt_2),935). orientation(lbpugt_4,135,935). coords(lbpugt_4,219,34,935).  visible(lbpugt_4,935). happensAt(inactive(lbpugt_4),935).
orientation(lbpugt_2,126,936). coords(lbpugt_2,199,96,936).  visible(lbpugt_2,936). happensAt(walking(lbpugt_2),936). orientation(lbpugt_4,135,936). coords(lbpugt_4,219,34,936).  visible(lbpugt_4,936). happensAt(inactive(lbpugt_4),936).
orientation(lbpugt_2,126,937). coords(lbpugt_2,197,93,937).  visible(lbpugt_2,937). happensAt(walking(lbpugt_2),937). orientation(lbpugt_4,135,937). coords(lbpugt_4,219,34,937).  visible(lbpugt_4,937). happensAt(inactive(lbpugt_4),937).
orientation(lbpugt_2,126,938). coords(lbpugt_2,196,91,938).  visible(lbpugt_2,938). happensAt(walking(lbpugt_2),938). orientation(lbpugt_4,135,938). coords(lbpugt_4,219,34,938).  visible(lbpugt_4,938). happensAt(inactive(lbpugt_4),938).
orientation(lbpugt_2,126,939). coords(lbpugt_2,195,89,939).  visible(lbpugt_2,939). happensAt(walking(lbpugt_2),939). orientation(lbpugt_4,135,939). coords(lbpugt_4,219,34,939).  visible(lbpugt_4,939). happensAt(inactive(lbpugt_4),939).
orientation(lbpugt_2,126,940). coords(lbpugt_2,195,87,940).  visible(lbpugt_2,940). happensAt(walking(lbpugt_2),940). orientation(lbpugt_4,135,940). coords(lbpugt_4,219,34,940).  visible(lbpugt_4,940). happensAt(inactive(lbpugt_4),940).
orientation(lbpugt_2,126,941). coords(lbpugt_2,195,86,941).  visible(lbpugt_2,941). happensAt(walking(lbpugt_2),941). orientation(lbpugt_4,135,941). coords(lbpugt_4,219,34,941).  visible(lbpugt_4,941). happensAt(inactive(lbpugt_4),941).
orientation(lbpugt_2,126,942). coords(lbpugt_2,194,85,942).  visible(lbpugt_2,942). happensAt(walking(lbpugt_2),942). orientation(lbpugt_4,135,942). coords(lbpugt_4,219,34,942).  visible(lbpugt_4,942). happensAt(inactive(lbpugt_4),942).
orientation(lbpugt_2,126,943). coords(lbpugt_2,194,84,943).  visible(lbpugt_2,943). happensAt(walking(lbpugt_2),943). orientation(lbpugt_4,135,943). coords(lbpugt_4,219,34,943).  visible(lbpugt_4,943). happensAt(inactive(lbpugt_4),943).
orientation(lbpugt_2,126,944). coords(lbpugt_2,194,84,944).  visible(lbpugt_2,944). happensAt(walking(lbpugt_2),944). orientation(lbpugt_4,135,944). coords(lbpugt_4,219,34,944).  visible(lbpugt_4,944). happensAt(inactive(lbpugt_4),944).
orientation(lbpugt_2,126,945). coords(lbpugt_2,194,83,945).  visible(lbpugt_2,945). happensAt(walking(lbpugt_2),945). orientation(lbpugt_4,135,945). coords(lbpugt_4,219,34,945).  visible(lbpugt_4,945). happensAt(inactive(lbpugt_4),945).
orientation(lbpugt_2,126,946). coords(lbpugt_2,194,82,946).  visible(lbpugt_2,946). happensAt(walking(lbpugt_2),946). orientation(lbpugt_4,135,946). coords(lbpugt_4,219,34,946).  visible(lbpugt_4,946). happensAt(inactive(lbpugt_4),946).
orientation(lbpugt_2,126,947). coords(lbpugt_2,194,82,947).  visible(lbpugt_2,947). happensAt(walking(lbpugt_2),947). orientation(lbpugt_4,135,947). coords(lbpugt_4,219,34,947).  visible(lbpugt_4,947). happensAt(inactive(lbpugt_4),947).
#show holdsAt/2.