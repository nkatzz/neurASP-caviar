
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

	orientation(lbbcgt_3,163,953). coords(lbbcgt_3,35,190,953).  visible(lbbcgt_3,953). happensAt(walking(lbbcgt_3),953). orientation(lbbcgt_6,151,953). coords(lbbcgt_6,85,86,953).  visible(lbbcgt_6,953). happensAt(walking(lbbcgt_6),953).
orientation(lbbcgt_3,163,954). coords(lbbcgt_3,35,190,954).  visible(lbbcgt_3,954). happensAt(walking(lbbcgt_3),954). orientation(lbbcgt_6,151,954). coords(lbbcgt_6,84,86,954).  visible(lbbcgt_6,954). happensAt(walking(lbbcgt_6),954).
orientation(lbbcgt_3,163,955). coords(lbbcgt_3,35,190,955).  visible(lbbcgt_3,955). happensAt(walking(lbbcgt_3),955). orientation(lbbcgt_6,151,955). coords(lbbcgt_6,83,86,955).  visible(lbbcgt_6,955). happensAt(walking(lbbcgt_6),955).
orientation(lbbcgt_3,163,956). coords(lbbcgt_3,35,191,956).  visible(lbbcgt_3,956). happensAt(walking(lbbcgt_3),956). orientation(lbbcgt_6,151,956). coords(lbbcgt_6,82,86,956).  visible(lbbcgt_6,956). happensAt(walking(lbbcgt_6),956).
orientation(lbbcgt_3,163,957). coords(lbbcgt_3,35,191,957).  visible(lbbcgt_3,957). happensAt(walking(lbbcgt_3),957). orientation(lbbcgt_6,151,957). coords(lbbcgt_6,81,86,957).  visible(lbbcgt_6,957). happensAt(walking(lbbcgt_6),957).
orientation(lbbcgt_3,163,958). coords(lbbcgt_3,35,192,958).  visible(lbbcgt_3,958). happensAt(walking(lbbcgt_3),958). orientation(lbbcgt_6,151,958). coords(lbbcgt_6,81,86,958).  visible(lbbcgt_6,958). happensAt(walking(lbbcgt_6),958).
orientation(lbbcgt_3,163,959). coords(lbbcgt_3,35,193,959).  visible(lbbcgt_3,959). happensAt(walking(lbbcgt_3),959). orientation(lbbcgt_6,151,959). coords(lbbcgt_6,81,86,959).  visible(lbbcgt_6,959). happensAt(walking(lbbcgt_6),959).
orientation(lbbcgt_3,163,960). coords(lbbcgt_3,35,195,960).  visible(lbbcgt_3,960). happensAt(walking(lbbcgt_3),960). orientation(lbbcgt_6,151,960). coords(lbbcgt_6,81,87,960).  visible(lbbcgt_6,960). happensAt(walking(lbbcgt_6),960).
orientation(lbbcgt_3,163,961). coords(lbbcgt_3,35,197,961).  visible(lbbcgt_3,961). happensAt(walking(lbbcgt_3),961). orientation(lbbcgt_6,151,961). coords(lbbcgt_6,81,87,961).  visible(lbbcgt_6,961). happensAt(walking(lbbcgt_6),961).
orientation(lbbcgt_3,163,962). coords(lbbcgt_3,35,197,962).  visible(lbbcgt_3,962). happensAt(walking(lbbcgt_3),962). orientation(lbbcgt_6,151,962). coords(lbbcgt_6,81,87,962).  visible(lbbcgt_6,962). happensAt(walking(lbbcgt_6),962).
orientation(lbbcgt_3,163,963). coords(lbbcgt_3,35,198,963).  visible(lbbcgt_3,963). happensAt(walking(lbbcgt_3),963). orientation(lbbcgt_6,151,963). coords(lbbcgt_6,81,87,963).  visible(lbbcgt_6,963). happensAt(walking(lbbcgt_6),963).
orientation(lbbcgt_3,163,964). coords(lbbcgt_3,35,198,964).  visible(lbbcgt_3,964). happensAt(walking(lbbcgt_3),964). orientation(lbbcgt_6,151,964). coords(lbbcgt_6,81,87,964).  visible(lbbcgt_6,964). happensAt(walking(lbbcgt_6),964).
orientation(lbbcgt_3,163,965). coords(lbbcgt_3,35,199,965).  visible(lbbcgt_3,965). happensAt(walking(lbbcgt_3),965). orientation(lbbcgt_6,151,965). coords(lbbcgt_6,79,87,965).  visible(lbbcgt_6,965). happensAt(walking(lbbcgt_6),965).
orientation(lbbcgt_3,163,966). coords(lbbcgt_3,35,199,966).  visible(lbbcgt_3,966). happensAt(walking(lbbcgt_3),966). orientation(lbbcgt_6,151,966). coords(lbbcgt_6,79,86,966).  visible(lbbcgt_6,966). happensAt(walking(lbbcgt_6),966).
orientation(lbbcgt_3,163,967). coords(lbbcgt_3,36,199,967).  visible(lbbcgt_3,967). happensAt(walking(lbbcgt_3),967). orientation(lbbcgt_6,151,967). coords(lbbcgt_6,78,87,967).  visible(lbbcgt_6,967). happensAt(walking(lbbcgt_6),967).
orientation(lbbcgt_3,163,968). coords(lbbcgt_3,36,200,968).  visible(lbbcgt_3,968). happensAt(walking(lbbcgt_3),968). orientation(lbbcgt_6,151,968). coords(lbbcgt_6,78,87,968).  visible(lbbcgt_6,968). happensAt(walking(lbbcgt_6),968).
orientation(lbbcgt_3,163,969). coords(lbbcgt_3,36,200,969).  visible(lbbcgt_3,969). happensAt(walking(lbbcgt_3),969). orientation(lbbcgt_6,151,969). coords(lbbcgt_6,77,87,969).  visible(lbbcgt_6,969). happensAt(walking(lbbcgt_6),969).
orientation(lbbcgt_3,163,970). coords(lbbcgt_3,35,201,970).  visible(lbbcgt_3,970). happensAt(walking(lbbcgt_3),970). orientation(lbbcgt_6,151,970). coords(lbbcgt_6,76,87,970).  visible(lbbcgt_6,970). happensAt(walking(lbbcgt_6),970).
orientation(lbbcgt_3,163,971). coords(lbbcgt_3,35,201,971).  visible(lbbcgt_3,971). happensAt(walking(lbbcgt_3),971). orientation(lbbcgt_6,151,971). coords(lbbcgt_6,76,87,971).  visible(lbbcgt_6,971). happensAt(walking(lbbcgt_6),971).
orientation(lbbcgt_3,163,972). coords(lbbcgt_3,35,202,972).  visible(lbbcgt_3,972). happensAt(walking(lbbcgt_3),972). orientation(lbbcgt_6,151,972). coords(lbbcgt_6,75,88,972).  visible(lbbcgt_6,972). happensAt(walking(lbbcgt_6),972).
orientation(lbbcgt_3,163,973). coords(lbbcgt_3,35,203,973).  visible(lbbcgt_3,973). happensAt(walking(lbbcgt_3),973). orientation(lbbcgt_6,151,973). coords(lbbcgt_6,76,88,973).  visible(lbbcgt_6,973). happensAt(walking(lbbcgt_6),973).
orientation(lbbcgt_3,163,974). coords(lbbcgt_3,35,204,974).  visible(lbbcgt_3,974). happensAt(walking(lbbcgt_3),974). orientation(lbbcgt_6,151,974). coords(lbbcgt_6,75,88,974).  visible(lbbcgt_6,974). happensAt(walking(lbbcgt_6),974).
orientation(lbbcgt_3,163,975). coords(lbbcgt_3,35,205,975).  visible(lbbcgt_3,975). happensAt(walking(lbbcgt_3),975). orientation(lbbcgt_6,151,975). coords(lbbcgt_6,75,88,975).  visible(lbbcgt_6,975). happensAt(walking(lbbcgt_6),975).
orientation(lbbcgt_3,163,976). coords(lbbcgt_3,35,205,976).  visible(lbbcgt_3,976). happensAt(walking(lbbcgt_3),976). orientation(lbbcgt_6,151,976). coords(lbbcgt_6,75,88,976).  visible(lbbcgt_6,976). happensAt(walking(lbbcgt_6),976).
orientation(lbbcgt_3,163,977). coords(lbbcgt_3,35,206,977).  visible(lbbcgt_3,977). happensAt(walking(lbbcgt_3),977). orientation(lbbcgt_6,151,977). coords(lbbcgt_6,75,89,977).  visible(lbbcgt_6,977). happensAt(walking(lbbcgt_6),977).
#show holdsAt/2.