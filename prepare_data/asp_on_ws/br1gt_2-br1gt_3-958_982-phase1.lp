
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

	orientation(br1gt_2,171,958). coords(br1gt_2,93,202,958).  visible(br1gt_2,958). happensAt(walking(br1gt_2),958). orientation(br1gt_3,89,958). coords(br1gt_3,306,93,958).  visible(br1gt_3,958). happensAt(active(br1gt_3),958).
orientation(br1gt_2,171,959). coords(br1gt_2,93,202,959).  visible(br1gt_2,959). happensAt(walking(br1gt_2),959). orientation(br1gt_3,89,959). coords(br1gt_3,306,93,959).  visible(br1gt_3,959). happensAt(active(br1gt_3),959).
orientation(br1gt_2,171,960). coords(br1gt_2,94,202,960).  visible(br1gt_2,960). happensAt(walking(br1gt_2),960). orientation(br1gt_3,89,960). coords(br1gt_3,306,93,960).  visible(br1gt_3,960). happensAt(active(br1gt_3),960).
orientation(br1gt_2,171,961). coords(br1gt_2,94,202,961).  visible(br1gt_2,961). happensAt(walking(br1gt_2),961). orientation(br1gt_3,89,961). coords(br1gt_3,306,93,961).  visible(br1gt_3,961). happensAt(active(br1gt_3),961).
orientation(br1gt_2,171,962). coords(br1gt_2,96,204,962).  visible(br1gt_2,962). happensAt(walking(br1gt_2),962). orientation(br1gt_3,89,962). coords(br1gt_3,306,93,962).  visible(br1gt_3,962). happensAt(active(br1gt_3),962).
orientation(br1gt_2,171,963). coords(br1gt_2,96,204,963).  visible(br1gt_2,963). happensAt(walking(br1gt_2),963). orientation(br1gt_3,89,963). coords(br1gt_3,305,93,963).  visible(br1gt_3,963). happensAt(active(br1gt_3),963).
orientation(br1gt_2,166,964). coords(br1gt_2,96,204,964).  visible(br1gt_2,964). happensAt(walking(br1gt_2),964). orientation(br1gt_3,89,964). coords(br1gt_3,304,93,964).  visible(br1gt_3,964). happensAt(active(br1gt_3),964).
orientation(br1gt_2,166,965). coords(br1gt_2,98,205,965).  visible(br1gt_2,965). happensAt(walking(br1gt_2),965). orientation(br1gt_3,89,965). coords(br1gt_3,304,93,965).  visible(br1gt_3,965). happensAt(active(br1gt_3),965).
orientation(br1gt_2,166,966). coords(br1gt_2,98,205,966).  visible(br1gt_2,966). happensAt(walking(br1gt_2),966). orientation(br1gt_3,89,966). coords(br1gt_3,304,92,966).  visible(br1gt_3,966). happensAt(active(br1gt_3),966).
orientation(br1gt_2,166,967). coords(br1gt_2,99,205,967).  visible(br1gt_2,967). happensAt(walking(br1gt_2),967). orientation(br1gt_3,89,967). coords(br1gt_3,303,92,967).  visible(br1gt_3,967). happensAt(active(br1gt_3),967).
orientation(br1gt_2,166,968). coords(br1gt_2,99,206,968).  visible(br1gt_2,968). happensAt(walking(br1gt_2),968). orientation(br1gt_3,89,968). coords(br1gt_3,302,91,968).  visible(br1gt_3,968). happensAt(walking(br1gt_3),968).
orientation(br1gt_2,166,969). coords(br1gt_2,99,206,969).  visible(br1gt_2,969). happensAt(walking(br1gt_2),969). orientation(br1gt_3,89,969). coords(br1gt_3,302,91,969).  visible(br1gt_3,969). happensAt(walking(br1gt_3),969).
orientation(br1gt_2,166,970). coords(br1gt_2,99,206,970).  visible(br1gt_2,970). happensAt(walking(br1gt_2),970). orientation(br1gt_3,89,970). coords(br1gt_3,302,91,970).  visible(br1gt_3,970). happensAt(walking(br1gt_3),970).
orientation(br1gt_2,166,971). coords(br1gt_2,99,206,971).  visible(br1gt_2,971). happensAt(walking(br1gt_2),971). orientation(br1gt_3,89,971). coords(br1gt_3,302,91,971).  visible(br1gt_3,971). happensAt(walking(br1gt_3),971).
orientation(br1gt_2,166,972). coords(br1gt_2,99,207,972).  visible(br1gt_2,972). happensAt(walking(br1gt_2),972). orientation(br1gt_3,89,972). coords(br1gt_3,302,91,972).  visible(br1gt_3,972). happensAt(walking(br1gt_3),972).
orientation(br1gt_2,166,973). coords(br1gt_2,100,207,973).  visible(br1gt_2,973). happensAt(walking(br1gt_2),973). orientation(br1gt_3,89,973). coords(br1gt_3,301,91,973).  visible(br1gt_3,973). happensAt(walking(br1gt_3),973).
orientation(br1gt_2,166,974). coords(br1gt_2,100,207,974).  visible(br1gt_2,974). happensAt(walking(br1gt_2),974). orientation(br1gt_3,89,974). coords(br1gt_3,301,91,974).  visible(br1gt_3,974). happensAt(walking(br1gt_3),974).
orientation(br1gt_2,166,975). coords(br1gt_2,100,207,975).  visible(br1gt_2,975). happensAt(walking(br1gt_2),975). orientation(br1gt_3,89,975). coords(br1gt_3,301,91,975).  visible(br1gt_3,975). happensAt(walking(br1gt_3),975).
orientation(br1gt_2,166,976). coords(br1gt_2,100,207,976).  visible(br1gt_2,976). happensAt(walking(br1gt_2),976). orientation(br1gt_3,104,976). coords(br1gt_3,299,91,976).  visible(br1gt_3,976). happensAt(walking(br1gt_3),976).
orientation(br1gt_2,166,977). coords(br1gt_2,102,207,977).  visible(br1gt_2,977). happensAt(walking(br1gt_2),977). orientation(br1gt_3,112,977). coords(br1gt_3,299,91,977).  visible(br1gt_3,977). happensAt(walking(br1gt_3),977).
orientation(br1gt_2,166,978). coords(br1gt_2,103,208,978).  visible(br1gt_2,978). happensAt(walking(br1gt_2),978). orientation(br1gt_3,112,978). coords(br1gt_3,298,91,978).  visible(br1gt_3,978). happensAt(walking(br1gt_3),978).
orientation(br1gt_2,166,979). coords(br1gt_2,103,208,979).  visible(br1gt_2,979). happensAt(walking(br1gt_2),979). orientation(br1gt_3,112,979). coords(br1gt_3,297,91,979).  visible(br1gt_3,979). happensAt(walking(br1gt_3),979).
orientation(br1gt_2,166,980). coords(br1gt_2,104,208,980).  visible(br1gt_2,980). happensAt(walking(br1gt_2),980). orientation(br1gt_3,117,980). coords(br1gt_3,297,91,980).  visible(br1gt_3,980). happensAt(walking(br1gt_3),980).
orientation(br1gt_2,166,981). coords(br1gt_2,105,208,981).  visible(br1gt_2,981). happensAt(walking(br1gt_2),981). orientation(br1gt_3,122,981). coords(br1gt_3,294,93,981).  visible(br1gt_3,981). happensAt(walking(br1gt_3),981).
orientation(br1gt_2,166,982). coords(br1gt_2,105,208,982).  visible(br1gt_2,982). happensAt(walking(br1gt_2),982). orientation(br1gt_3,122,982). coords(br1gt_3,292,94,982).  visible(br1gt_3,982). happensAt(walking(br1gt_3),982).
#show holdsAt/2.