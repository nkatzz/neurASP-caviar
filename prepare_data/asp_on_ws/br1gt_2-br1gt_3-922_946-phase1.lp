
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

	orientation(br1gt_2,2,922). coords(br1gt_2,86,198,922).  visible(br1gt_2,922). happensAt(active(br1gt_2),922). orientation(br1gt_3,89,922). coords(br1gt_3,297,98,922).  visible(br1gt_3,922). happensAt(active(br1gt_3),922).
orientation(br1gt_2,177,923). coords(br1gt_2,86,198,923).  visible(br1gt_2,923). happensAt(active(br1gt_2),923). orientation(br1gt_3,89,923). coords(br1gt_3,298,96,923).  visible(br1gt_3,923). happensAt(active(br1gt_3),923).
orientation(br1gt_2,177,924). coords(br1gt_2,86,198,924).  visible(br1gt_2,924). happensAt(active(br1gt_2),924). orientation(br1gt_3,89,924). coords(br1gt_3,299,96,924).  visible(br1gt_3,924). happensAt(active(br1gt_3),924).
orientation(br1gt_2,177,925). coords(br1gt_2,86,198,925).  visible(br1gt_2,925). happensAt(active(br1gt_2),925). orientation(br1gt_3,89,925). coords(br1gt_3,299,96,925).  visible(br1gt_3,925). happensAt(active(br1gt_3),925).
orientation(br1gt_2,174,926). coords(br1gt_2,86,197,926).  visible(br1gt_2,926). happensAt(active(br1gt_2),926). orientation(br1gt_3,89,926). coords(br1gt_3,299,95,926).  visible(br1gt_3,926). happensAt(active(br1gt_3),926).
orientation(br1gt_2,174,927). coords(br1gt_2,86,196,927).  visible(br1gt_2,927). happensAt(active(br1gt_2),927). orientation(br1gt_3,89,927). coords(br1gt_3,299,95,927).  visible(br1gt_3,927). happensAt(active(br1gt_3),927).
orientation(br1gt_2,171,928). coords(br1gt_2,86,198,928).  visible(br1gt_2,928). happensAt(active(br1gt_2),928). orientation(br1gt_3,89,928). coords(br1gt_3,299,95,928).  visible(br1gt_3,928). happensAt(active(br1gt_3),928).
orientation(br1gt_2,171,929). coords(br1gt_2,86,198,929).  visible(br1gt_2,929). happensAt(active(br1gt_2),929). orientation(br1gt_3,89,929). coords(br1gt_3,300,95,929).  visible(br1gt_3,929). happensAt(active(br1gt_3),929).
orientation(br1gt_2,171,930). coords(br1gt_2,86,198,930).  visible(br1gt_2,930). happensAt(walking(br1gt_2),930). orientation(br1gt_3,89,930). coords(br1gt_3,300,95,930).  visible(br1gt_3,930). happensAt(active(br1gt_3),930).
orientation(br1gt_2,171,931). coords(br1gt_2,86,198,931).  visible(br1gt_2,931). happensAt(walking(br1gt_2),931). orientation(br1gt_3,89,931). coords(br1gt_3,301,95,931).  visible(br1gt_3,931). happensAt(active(br1gt_3),931).
orientation(br1gt_2,171,932). coords(br1gt_2,86,198,932).  visible(br1gt_2,932). happensAt(walking(br1gt_2),932). orientation(br1gt_3,89,932). coords(br1gt_3,301,95,932).  visible(br1gt_3,932). happensAt(active(br1gt_3),932).
orientation(br1gt_2,171,933). coords(br1gt_2,86,198,933).  visible(br1gt_2,933). happensAt(walking(br1gt_2),933). orientation(br1gt_3,89,933). coords(br1gt_3,301,95,933).  visible(br1gt_3,933). happensAt(active(br1gt_3),933).
orientation(br1gt_2,171,934). coords(br1gt_2,86,199,934).  visible(br1gt_2,934). happensAt(walking(br1gt_2),934). orientation(br1gt_3,89,934). coords(br1gt_3,301,94,934).  visible(br1gt_3,934). happensAt(active(br1gt_3),934).
orientation(br1gt_2,171,935). coords(br1gt_2,86,199,935).  visible(br1gt_2,935). happensAt(walking(br1gt_2),935). orientation(br1gt_3,89,935). coords(br1gt_3,303,95,935).  visible(br1gt_3,935). happensAt(active(br1gt_3),935).
orientation(br1gt_2,171,936). coords(br1gt_2,86,199,936).  visible(br1gt_2,936). happensAt(walking(br1gt_2),936). orientation(br1gt_3,89,936). coords(br1gt_3,303,94,936).  visible(br1gt_3,936). happensAt(active(br1gt_3),936).
orientation(br1gt_2,171,937). coords(br1gt_2,86,199,937).  visible(br1gt_2,937). happensAt(walking(br1gt_2),937). orientation(br1gt_3,89,937). coords(br1gt_3,304,94,937).  visible(br1gt_3,937). happensAt(active(br1gt_3),937).
orientation(br1gt_2,171,938). coords(br1gt_2,86,199,938).  visible(br1gt_2,938). happensAt(walking(br1gt_2),938). orientation(br1gt_3,89,938). coords(br1gt_3,304,94,938).  visible(br1gt_3,938). happensAt(active(br1gt_3),938).
orientation(br1gt_2,171,939). coords(br1gt_2,87,199,939).  visible(br1gt_2,939). happensAt(walking(br1gt_2),939). orientation(br1gt_3,89,939). coords(br1gt_3,304,94,939).  visible(br1gt_3,939). happensAt(active(br1gt_3),939).
orientation(br1gt_2,171,940). coords(br1gt_2,88,199,940).  visible(br1gt_2,940). happensAt(walking(br1gt_2),940). orientation(br1gt_3,89,940). coords(br1gt_3,304,94,940).  visible(br1gt_3,940). happensAt(active(br1gt_3),940).
orientation(br1gt_2,171,941). coords(br1gt_2,88,200,941).  visible(br1gt_2,941). happensAt(walking(br1gt_2),941). orientation(br1gt_3,89,941). coords(br1gt_3,304,94,941).  visible(br1gt_3,941). happensAt(active(br1gt_3),941).
orientation(br1gt_2,171,942). coords(br1gt_2,89,200,942).  visible(br1gt_2,942). happensAt(walking(br1gt_2),942). orientation(br1gt_3,89,942). coords(br1gt_3,304,94,942).  visible(br1gt_3,942). happensAt(active(br1gt_3),942).
orientation(br1gt_2,171,943). coords(br1gt_2,90,201,943).  visible(br1gt_2,943). happensAt(walking(br1gt_2),943). orientation(br1gt_3,89,943). coords(br1gt_3,306,94,943).  visible(br1gt_3,943). happensAt(active(br1gt_3),943).
orientation(br1gt_2,171,944). coords(br1gt_2,90,200,944).  visible(br1gt_2,944). happensAt(walking(br1gt_2),944). orientation(br1gt_3,89,944). coords(br1gt_3,306,94,944).  visible(br1gt_3,944). happensAt(active(br1gt_3),944).
orientation(br1gt_2,171,945). coords(br1gt_2,90,200,945).  visible(br1gt_2,945). happensAt(walking(br1gt_2),945). orientation(br1gt_3,89,945). coords(br1gt_3,306,94,945).  visible(br1gt_3,945). happensAt(active(br1gt_3),945).
orientation(br1gt_2,171,946). coords(br1gt_2,91,200,946).  visible(br1gt_2,946). happensAt(walking(br1gt_2),946). orientation(br1gt_3,89,946). coords(br1gt_3,306,94,946).  visible(br1gt_3,946). happensAt(active(br1gt_3),946).
#show holdsAt/2.