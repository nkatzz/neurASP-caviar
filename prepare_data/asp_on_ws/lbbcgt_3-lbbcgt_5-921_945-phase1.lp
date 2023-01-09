
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

	orientation(lbbcgt_3,163,921). coords(lbbcgt_3,37,168,921).  visible(lbbcgt_3,921). happensAt(walking(lbbcgt_3),921). orientation(lbbcgt_5,167,921). coords(lbbcgt_5,212,213,921).  visible(lbbcgt_5,921). happensAt(walking(lbbcgt_5),921).
orientation(lbbcgt_3,163,922). coords(lbbcgt_3,37,169,922).  visible(lbbcgt_3,922). happensAt(walking(lbbcgt_3),922). orientation(lbbcgt_5,167,922). coords(lbbcgt_5,211,213,922).  visible(lbbcgt_5,922). happensAt(walking(lbbcgt_5),922).
orientation(lbbcgt_3,163,923). coords(lbbcgt_3,37,169,923).  visible(lbbcgt_3,923). happensAt(walking(lbbcgt_3),923). orientation(lbbcgt_5,167,923). coords(lbbcgt_5,210,212,923).  visible(lbbcgt_5,923). happensAt(walking(lbbcgt_5),923).
orientation(lbbcgt_3,163,924). coords(lbbcgt_3,37,170,924).  visible(lbbcgt_3,924). happensAt(walking(lbbcgt_3),924). orientation(lbbcgt_5,164,924). coords(lbbcgt_5,209,212,924).  visible(lbbcgt_5,924). happensAt(walking(lbbcgt_5),924).
orientation(lbbcgt_3,163,925). coords(lbbcgt_3,37,170,925).  visible(lbbcgt_3,925). happensAt(walking(lbbcgt_3),925). orientation(lbbcgt_5,164,925). coords(lbbcgt_5,208,210,925).  visible(lbbcgt_5,925). happensAt(walking(lbbcgt_5),925).
orientation(lbbcgt_3,163,926). coords(lbbcgt_3,36,170,926).  visible(lbbcgt_3,926). happensAt(walking(lbbcgt_3),926). orientation(lbbcgt_5,164,926). coords(lbbcgt_5,207,207,926).  visible(lbbcgt_5,926). happensAt(walking(lbbcgt_5),926).
orientation(lbbcgt_3,163,927). coords(lbbcgt_3,36,170,927).  visible(lbbcgt_3,927). happensAt(walking(lbbcgt_3),927). orientation(lbbcgt_5,164,927). coords(lbbcgt_5,206,204,927).  visible(lbbcgt_5,927). happensAt(walking(lbbcgt_5),927).
orientation(lbbcgt_3,163,928). coords(lbbcgt_3,36,172,928).  visible(lbbcgt_3,928). happensAt(walking(lbbcgt_3),928). orientation(lbbcgt_5,158,928). coords(lbbcgt_5,206,201,928).  visible(lbbcgt_5,928). happensAt(walking(lbbcgt_5),928).
orientation(lbbcgt_3,163,929). coords(lbbcgt_3,36,172,929).  visible(lbbcgt_3,929). happensAt(walking(lbbcgt_3),929). orientation(lbbcgt_5,158,929). coords(lbbcgt_5,205,198,929).  visible(lbbcgt_5,929). happensAt(walking(lbbcgt_5),929).
orientation(lbbcgt_3,163,930). coords(lbbcgt_3,36,173,930).  visible(lbbcgt_3,930). happensAt(walking(lbbcgt_3),930). orientation(lbbcgt_5,158,930). coords(lbbcgt_5,205,193,930).  visible(lbbcgt_5,930). happensAt(walking(lbbcgt_5),930).
orientation(lbbcgt_3,163,931). coords(lbbcgt_3,35,173,931).  visible(lbbcgt_3,931). happensAt(walking(lbbcgt_3),931). orientation(lbbcgt_5,158,931). coords(lbbcgt_5,204,190,931).  visible(lbbcgt_5,931). happensAt(walking(lbbcgt_5),931).
orientation(lbbcgt_3,163,932). coords(lbbcgt_3,35,176,932).  visible(lbbcgt_3,932). happensAt(walking(lbbcgt_3),932). orientation(lbbcgt_5,158,932). coords(lbbcgt_5,204,189,932).  visible(lbbcgt_5,932). happensAt(walking(lbbcgt_5),932).
orientation(lbbcgt_3,163,933). coords(lbbcgt_3,35,176,933).  visible(lbbcgt_3,933). happensAt(walking(lbbcgt_3),933). orientation(lbbcgt_5,158,933). coords(lbbcgt_5,203,187,933).  visible(lbbcgt_5,933). happensAt(walking(lbbcgt_5),933).
orientation(lbbcgt_3,163,934). coords(lbbcgt_3,35,178,934).  visible(lbbcgt_3,934). happensAt(walking(lbbcgt_3),934). orientation(lbbcgt_5,158,934). coords(lbbcgt_5,203,186,934).  visible(lbbcgt_5,934). happensAt(walking(lbbcgt_5),934).
orientation(lbbcgt_3,163,935). coords(lbbcgt_3,35,178,935).  visible(lbbcgt_3,935). happensAt(walking(lbbcgt_3),935). orientation(lbbcgt_5,158,935). coords(lbbcgt_5,203,185,935).  visible(lbbcgt_5,935). happensAt(walking(lbbcgt_5),935).
orientation(lbbcgt_3,163,936). coords(lbbcgt_3,35,179,936).  visible(lbbcgt_3,936). happensAt(walking(lbbcgt_3),936). orientation(lbbcgt_5,158,936). coords(lbbcgt_5,202,184,936).  visible(lbbcgt_5,936). happensAt(walking(lbbcgt_5),936).
orientation(lbbcgt_3,163,937). coords(lbbcgt_3,35,179,937).  visible(lbbcgt_3,937). happensAt(walking(lbbcgt_3),937). orientation(lbbcgt_5,158,937). coords(lbbcgt_5,202,183,937).  visible(lbbcgt_5,937). happensAt(walking(lbbcgt_5),937).
orientation(lbbcgt_3,163,938). coords(lbbcgt_3,35,179,938).  visible(lbbcgt_3,938). happensAt(walking(lbbcgt_3),938). orientation(lbbcgt_5,158,938). coords(lbbcgt_5,201,182,938).  visible(lbbcgt_5,938). happensAt(walking(lbbcgt_5),938).
orientation(lbbcgt_3,163,939). coords(lbbcgt_3,36,179,939).  visible(lbbcgt_3,939). happensAt(walking(lbbcgt_3),939). orientation(lbbcgt_5,158,939). coords(lbbcgt_5,201,180,939).  visible(lbbcgt_5,939). happensAt(walking(lbbcgt_5),939).
orientation(lbbcgt_3,163,940). coords(lbbcgt_3,36,179,940).  visible(lbbcgt_3,940). happensAt(walking(lbbcgt_3),940). orientation(lbbcgt_5,158,940). coords(lbbcgt_5,199,177,940).  visible(lbbcgt_5,940). happensAt(walking(lbbcgt_5),940).
orientation(lbbcgt_3,163,941). coords(lbbcgt_3,35,180,941).  visible(lbbcgt_3,941). happensAt(walking(lbbcgt_3),941). orientation(lbbcgt_5,158,941). coords(lbbcgt_5,198,175,941).  visible(lbbcgt_5,941). happensAt(walking(lbbcgt_5),941).
orientation(lbbcgt_3,163,942). coords(lbbcgt_3,35,180,942).  visible(lbbcgt_3,942). happensAt(walking(lbbcgt_3),942). orientation(lbbcgt_5,158,942). coords(lbbcgt_5,197,173,942).  visible(lbbcgt_5,942). happensAt(walking(lbbcgt_5),942).
orientation(lbbcgt_3,163,943). coords(lbbcgt_3,35,181,943).  visible(lbbcgt_3,943). happensAt(walking(lbbcgt_3),943). orientation(lbbcgt_5,158,943). coords(lbbcgt_5,195,172,943).  visible(lbbcgt_5,943). happensAt(walking(lbbcgt_5),943).
orientation(lbbcgt_3,163,944). coords(lbbcgt_3,35,182,944).  visible(lbbcgt_3,944). happensAt(walking(lbbcgt_3),944). orientation(lbbcgt_5,158,944). coords(lbbcgt_5,193,171,944).  visible(lbbcgt_5,944). happensAt(walking(lbbcgt_5),944).
orientation(lbbcgt_3,163,945). coords(lbbcgt_3,35,184,945).  visible(lbbcgt_3,945). happensAt(walking(lbbcgt_3),945). orientation(lbbcgt_5,158,945). coords(lbbcgt_5,192,169,945).  visible(lbbcgt_5,945). happensAt(walking(lbbcgt_5),945).
#show holdsAt/2.