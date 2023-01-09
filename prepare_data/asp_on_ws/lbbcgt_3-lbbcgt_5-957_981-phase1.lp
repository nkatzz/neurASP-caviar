
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

	orientation(lbbcgt_3,163,957). coords(lbbcgt_3,35,191,957).  visible(lbbcgt_3,957). happensAt(walking(lbbcgt_3),957). orientation(lbbcgt_5,158,957). coords(lbbcgt_5,188,155,957).  visible(lbbcgt_5,957). happensAt(walking(lbbcgt_5),957).
orientation(lbbcgt_3,163,958). coords(lbbcgt_3,35,192,958).  visible(lbbcgt_3,958). happensAt(walking(lbbcgt_3),958). orientation(lbbcgt_5,158,958). coords(lbbcgt_5,187,154,958).  visible(lbbcgt_5,958). happensAt(walking(lbbcgt_5),958).
orientation(lbbcgt_3,163,959). coords(lbbcgt_3,35,193,959).  visible(lbbcgt_3,959). happensAt(walking(lbbcgt_3),959). orientation(lbbcgt_5,158,959). coords(lbbcgt_5,187,153,959).  visible(lbbcgt_5,959). happensAt(walking(lbbcgt_5),959).
orientation(lbbcgt_3,163,960). coords(lbbcgt_3,35,195,960).  visible(lbbcgt_3,960). happensAt(walking(lbbcgt_3),960). orientation(lbbcgt_5,158,960). coords(lbbcgt_5,187,151,960).  visible(lbbcgt_5,960). happensAt(walking(lbbcgt_5),960).
orientation(lbbcgt_3,163,961). coords(lbbcgt_3,35,197,961).  visible(lbbcgt_3,961). happensAt(walking(lbbcgt_3),961). orientation(lbbcgt_5,158,961). coords(lbbcgt_5,186,150,961).  visible(lbbcgt_5,961). happensAt(walking(lbbcgt_5),961).
orientation(lbbcgt_3,163,962). coords(lbbcgt_3,35,197,962).  visible(lbbcgt_3,962). happensAt(walking(lbbcgt_3),962). orientation(lbbcgt_5,158,962). coords(lbbcgt_5,186,150,962).  visible(lbbcgt_5,962). happensAt(walking(lbbcgt_5),962).
orientation(lbbcgt_3,163,963). coords(lbbcgt_3,35,198,963).  visible(lbbcgt_3,963). happensAt(walking(lbbcgt_3),963). orientation(lbbcgt_5,158,963). coords(lbbcgt_5,186,148,963).  visible(lbbcgt_5,963). happensAt(walking(lbbcgt_5),963).
orientation(lbbcgt_3,163,964). coords(lbbcgt_3,35,198,964).  visible(lbbcgt_3,964). happensAt(walking(lbbcgt_3),964). orientation(lbbcgt_5,158,964). coords(lbbcgt_5,186,147,964).  visible(lbbcgt_5,964). happensAt(walking(lbbcgt_5),964).
orientation(lbbcgt_3,163,965). coords(lbbcgt_3,35,199,965).  visible(lbbcgt_3,965). happensAt(walking(lbbcgt_3),965). orientation(lbbcgt_5,158,965). coords(lbbcgt_5,186,144,965).  visible(lbbcgt_5,965). happensAt(walking(lbbcgt_5),965).
orientation(lbbcgt_3,163,966). coords(lbbcgt_3,35,199,966).  visible(lbbcgt_3,966). happensAt(walking(lbbcgt_3),966). orientation(lbbcgt_5,158,966). coords(lbbcgt_5,187,143,966).  visible(lbbcgt_5,966). happensAt(walking(lbbcgt_5),966).
orientation(lbbcgt_3,163,967). coords(lbbcgt_3,36,199,967).  visible(lbbcgt_3,967). happensAt(walking(lbbcgt_3),967). orientation(lbbcgt_5,158,967). coords(lbbcgt_5,187,142,967).  visible(lbbcgt_5,967). happensAt(walking(lbbcgt_5),967).
orientation(lbbcgt_3,163,968). coords(lbbcgt_3,36,200,968).  visible(lbbcgt_3,968). happensAt(walking(lbbcgt_3),968). orientation(lbbcgt_5,154,968). coords(lbbcgt_5,187,142,968).  visible(lbbcgt_5,968). happensAt(walking(lbbcgt_5),968).
orientation(lbbcgt_3,163,969). coords(lbbcgt_3,36,200,969).  visible(lbbcgt_3,969). happensAt(walking(lbbcgt_3),969). orientation(lbbcgt_5,154,969). coords(lbbcgt_5,187,142,969).  visible(lbbcgt_5,969). happensAt(walking(lbbcgt_5),969).
orientation(lbbcgt_3,163,970). coords(lbbcgt_3,35,201,970).  visible(lbbcgt_3,970). happensAt(walking(lbbcgt_3),970). orientation(lbbcgt_5,154,970). coords(lbbcgt_5,187,141,970).  visible(lbbcgt_5,970). happensAt(walking(lbbcgt_5),970).
orientation(lbbcgt_3,163,971). coords(lbbcgt_3,35,201,971).  visible(lbbcgt_3,971). happensAt(walking(lbbcgt_3),971). orientation(lbbcgt_5,154,971). coords(lbbcgt_5,188,139,971).  visible(lbbcgt_5,971). happensAt(walking(lbbcgt_5),971).
orientation(lbbcgt_3,163,972). coords(lbbcgt_3,35,202,972).  visible(lbbcgt_3,972). happensAt(walking(lbbcgt_3),972). orientation(lbbcgt_5,154,972). coords(lbbcgt_5,188,137,972).  visible(lbbcgt_5,972). happensAt(walking(lbbcgt_5),972).
orientation(lbbcgt_3,163,973). coords(lbbcgt_3,35,203,973).  visible(lbbcgt_3,973). happensAt(walking(lbbcgt_3),973). orientation(lbbcgt_5,154,973). coords(lbbcgt_5,188,135,973).  visible(lbbcgt_5,973). happensAt(walking(lbbcgt_5),973).
orientation(lbbcgt_3,163,974). coords(lbbcgt_3,35,204,974).  visible(lbbcgt_3,974). happensAt(walking(lbbcgt_3),974). orientation(lbbcgt_5,154,974). coords(lbbcgt_5,188,133,974).  visible(lbbcgt_5,974). happensAt(walking(lbbcgt_5),974).
orientation(lbbcgt_3,163,975). coords(lbbcgt_3,35,205,975).  visible(lbbcgt_3,975). happensAt(walking(lbbcgt_3),975). orientation(lbbcgt_5,154,975). coords(lbbcgt_5,188,132,975).  visible(lbbcgt_5,975). happensAt(walking(lbbcgt_5),975).
orientation(lbbcgt_3,163,976). coords(lbbcgt_3,35,205,976).  visible(lbbcgt_3,976). happensAt(walking(lbbcgt_3),976). orientation(lbbcgt_5,154,976). coords(lbbcgt_5,188,132,976).  visible(lbbcgt_5,976). happensAt(walking(lbbcgt_5),976).
orientation(lbbcgt_3,163,977). coords(lbbcgt_3,35,206,977).  visible(lbbcgt_3,977). happensAt(walking(lbbcgt_3),977). orientation(lbbcgt_5,154,977). coords(lbbcgt_5,188,131,977).  visible(lbbcgt_5,977). happensAt(walking(lbbcgt_5),977).
orientation(lbbcgt_3,163,978). coords(lbbcgt_3,35,207,978).  visible(lbbcgt_3,978). happensAt(walking(lbbcgt_3),978). orientation(lbbcgt_5,154,978). coords(lbbcgt_5,188,131,978).  visible(lbbcgt_5,978). happensAt(walking(lbbcgt_5),978).
orientation(lbbcgt_3,163,979). coords(lbbcgt_3,35,207,979).  visible(lbbcgt_3,979). happensAt(walking(lbbcgt_3),979). orientation(lbbcgt_5,154,979). coords(lbbcgt_5,188,130,979).  visible(lbbcgt_5,979). happensAt(walking(lbbcgt_5),979).
orientation(lbbcgt_3,163,980). coords(lbbcgt_3,35,207,980).  visible(lbbcgt_3,980). happensAt(walking(lbbcgt_3),980). orientation(lbbcgt_5,154,980). coords(lbbcgt_5,188,130,980).  visible(lbbcgt_5,980). happensAt(walking(lbbcgt_5),980).
orientation(lbbcgt_3,163,981). coords(lbbcgt_3,31,207,981).  visible(lbbcgt_3,981). happensAt(walking(lbbcgt_3),981). orientation(lbbcgt_5,154,981). coords(lbbcgt_5,188,129,981).  visible(lbbcgt_5,981). happensAt(walking(lbbcgt_5),981).
#show holdsAt/2.