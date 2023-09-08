
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

	orientation(wk2gt_1,133,956). coords(wk2gt_1,123,46,956).  visible(wk2gt_1,956). happensAt(walking(wk2gt_1),956). orientation(wk2gt_5,150,956). coords(wk2gt_5,81,100,956).  visible(wk2gt_5,956). happensAt(walking(wk2gt_5),956).
orientation(wk2gt_1,133,957). coords(wk2gt_1,123,46,957).  visible(wk2gt_1,957). happensAt(walking(wk2gt_1),957). orientation(wk2gt_5,150,957). coords(wk2gt_5,82,101,957).  visible(wk2gt_5,957). happensAt(walking(wk2gt_5),957).
orientation(wk2gt_1,133,958). coords(wk2gt_1,124,46,958).  visible(wk2gt_1,958). happensAt(walking(wk2gt_1),958). orientation(wk2gt_5,150,958). coords(wk2gt_5,82,101,958).  visible(wk2gt_5,958). happensAt(walking(wk2gt_5),958).
orientation(wk2gt_1,133,959). coords(wk2gt_1,124,46,959).  visible(wk2gt_1,959). happensAt(walking(wk2gt_1),959). orientation(wk2gt_5,150,959). coords(wk2gt_5,82,101,959).  visible(wk2gt_5,959). happensAt(walking(wk2gt_5),959).
orientation(wk2gt_1,133,960). coords(wk2gt_1,125,45,960).  visible(wk2gt_1,960). happensAt(walking(wk2gt_1),960). orientation(wk2gt_5,150,960). coords(wk2gt_5,82,101,960).  visible(wk2gt_5,960). happensAt(walking(wk2gt_5),960).
orientation(wk2gt_1,133,961). coords(wk2gt_1,125,45,961).  visible(wk2gt_1,961). happensAt(walking(wk2gt_1),961). orientation(wk2gt_5,150,961). coords(wk2gt_5,82,102,961).  visible(wk2gt_5,961). happensAt(walking(wk2gt_5),961).
orientation(wk2gt_1,133,962). coords(wk2gt_1,125,45,962).  visible(wk2gt_1,962). happensAt(walking(wk2gt_1),962). orientation(wk2gt_5,150,962). coords(wk2gt_5,82,103,962).  visible(wk2gt_5,962). happensAt(walking(wk2gt_5),962).
orientation(wk2gt_1,133,963). coords(wk2gt_1,125,45,963).  visible(wk2gt_1,963). happensAt(walking(wk2gt_1),963). orientation(wk2gt_5,150,963). coords(wk2gt_5,82,104,963).  visible(wk2gt_5,963). happensAt(walking(wk2gt_5),963).
orientation(wk2gt_1,133,964). coords(wk2gt_1,125,45,964).  visible(wk2gt_1,964). happensAt(walking(wk2gt_1),964). orientation(wk2gt_5,150,964). coords(wk2gt_5,83,104,964).  visible(wk2gt_5,964). happensAt(walking(wk2gt_5),964).
orientation(wk2gt_1,133,965). coords(wk2gt_1,126,45,965).  visible(wk2gt_1,965). happensAt(walking(wk2gt_1),965). orientation(wk2gt_5,150,965). coords(wk2gt_5,84,105,965).  visible(wk2gt_5,965). happensAt(walking(wk2gt_5),965).
orientation(wk2gt_1,133,966). coords(wk2gt_1,127,45,966).  visible(wk2gt_1,966). happensAt(walking(wk2gt_1),966). orientation(wk2gt_5,150,966). coords(wk2gt_5,85,106,966).  visible(wk2gt_5,966). happensAt(walking(wk2gt_5),966).
orientation(wk2gt_1,133,967). coords(wk2gt_1,127,45,967).  visible(wk2gt_1,967). happensAt(walking(wk2gt_1),967). orientation(wk2gt_5,150,967). coords(wk2gt_5,85,106,967).  visible(wk2gt_5,967). happensAt(walking(wk2gt_5),967).
orientation(wk2gt_1,133,968). coords(wk2gt_1,127,45,968).  visible(wk2gt_1,968). happensAt(walking(wk2gt_1),968). orientation(wk2gt_5,150,968). coords(wk2gt_5,86,106,968).  visible(wk2gt_5,968). happensAt(walking(wk2gt_5),968).
orientation(wk2gt_1,133,969). coords(wk2gt_1,127,45,969).  visible(wk2gt_1,969). happensAt(walking(wk2gt_1),969). orientation(wk2gt_5,150,969). coords(wk2gt_5,86,106,969).  visible(wk2gt_5,969). happensAt(walking(wk2gt_5),969).
orientation(wk2gt_1,133,970). coords(wk2gt_1,128,44,970).  visible(wk2gt_1,970). happensAt(walking(wk2gt_1),970). orientation(wk2gt_5,150,970). coords(wk2gt_5,87,107,970).  visible(wk2gt_5,970). happensAt(walking(wk2gt_5),970).
orientation(wk2gt_1,133,971). coords(wk2gt_1,128,44,971).  visible(wk2gt_1,971). happensAt(walking(wk2gt_1),971). orientation(wk2gt_5,150,971). coords(wk2gt_5,87,107,971).  visible(wk2gt_5,971). happensAt(walking(wk2gt_5),971).
orientation(wk2gt_1,133,972). coords(wk2gt_1,128,44,972).  visible(wk2gt_1,972). happensAt(walking(wk2gt_1),972). orientation(wk2gt_5,150,972). coords(wk2gt_5,87,107,972).  visible(wk2gt_5,972). happensAt(walking(wk2gt_5),972).
orientation(wk2gt_1,133,973). coords(wk2gt_1,128,44,973).  visible(wk2gt_1,973). happensAt(walking(wk2gt_1),973). orientation(wk2gt_5,150,973). coords(wk2gt_5,87,107,973).  visible(wk2gt_5,973). happensAt(walking(wk2gt_5),973).
orientation(wk2gt_1,133,974). coords(wk2gt_1,128,43,974).  visible(wk2gt_1,974). happensAt(walking(wk2gt_1),974). orientation(wk2gt_5,150,974). coords(wk2gt_5,88,107,974).  visible(wk2gt_5,974). happensAt(walking(wk2gt_5),974).
orientation(wk2gt_1,133,975). coords(wk2gt_1,128,43,975).  visible(wk2gt_1,975). happensAt(walking(wk2gt_1),975). orientation(wk2gt_5,150,975). coords(wk2gt_5,90,108,975).  visible(wk2gt_5,975). happensAt(walking(wk2gt_5),975).
orientation(wk2gt_1,133,976). coords(wk2gt_1,129,43,976).  visible(wk2gt_1,976). happensAt(walking(wk2gt_1),976). orientation(wk2gt_5,150,976). coords(wk2gt_5,91,108,976).  visible(wk2gt_5,976). happensAt(walking(wk2gt_5),976).
orientation(wk2gt_1,133,977). coords(wk2gt_1,129,43,977).  visible(wk2gt_1,977). happensAt(walking(wk2gt_1),977). orientation(wk2gt_5,150,977). coords(wk2gt_5,92,108,977).  visible(wk2gt_5,977). happensAt(walking(wk2gt_5),977).
orientation(wk2gt_1,133,978). coords(wk2gt_1,129,43,978).  visible(wk2gt_1,978). happensAt(walking(wk2gt_1),978). orientation(wk2gt_5,150,978). coords(wk2gt_5,92,108,978).  visible(wk2gt_5,978). happensAt(walking(wk2gt_5),978).
orientation(wk2gt_1,133,979). coords(wk2gt_1,129,43,979).  visible(wk2gt_1,979). happensAt(walking(wk2gt_1),979). orientation(wk2gt_5,150,979). coords(wk2gt_5,93,108,979).  visible(wk2gt_5,979). happensAt(walking(wk2gt_5),979).
orientation(wk2gt_1,133,980). coords(wk2gt_1,130,43,980).  visible(wk2gt_1,980). happensAt(walking(wk2gt_1),980). orientation(wk2gt_5,150,980). coords(wk2gt_5,94,108,980).  visible(wk2gt_5,980). happensAt(walking(wk2gt_5),980).
#show holdsAt/2.