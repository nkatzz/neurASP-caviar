
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

	orientation(lb1gt_3,114,957). coords(lb1gt_3,287,94,957).  visible(lb1gt_3,957). happensAt(walking(lb1gt_3),957). orientation(lb1gt_4,90,957). coords(lb1gt_4,310,112,957).  visible(lb1gt_4,957). happensAt(inactive(lb1gt_4),957).
orientation(lb1gt_3,114,958). coords(lb1gt_3,286,94,958).  visible(lb1gt_3,958). happensAt(walking(lb1gt_3),958). orientation(lb1gt_4,90,958). coords(lb1gt_4,310,112,958).  visible(lb1gt_4,958). happensAt(inactive(lb1gt_4),958).
orientation(lb1gt_3,110,959). coords(lb1gt_3,286,93,959).  visible(lb1gt_3,959). happensAt(walking(lb1gt_3),959). orientation(lb1gt_4,90,959). coords(lb1gt_4,310,112,959).  visible(lb1gt_4,959). happensAt(inactive(lb1gt_4),959).
orientation(lb1gt_3,106,960). coords(lb1gt_3,286,92,960).  visible(lb1gt_3,960). happensAt(walking(lb1gt_3),960). orientation(lb1gt_4,90,960). coords(lb1gt_4,310,112,960).  visible(lb1gt_4,960). happensAt(inactive(lb1gt_4),960).
orientation(lb1gt_3,108,961). coords(lb1gt_3,284,91,961).  visible(lb1gt_3,961). happensAt(walking(lb1gt_3),961). orientation(lb1gt_4,90,961). coords(lb1gt_4,310,112,961).  visible(lb1gt_4,961). happensAt(inactive(lb1gt_4),961).
orientation(lb1gt_3,103,962). coords(lb1gt_3,283,90,962).  visible(lb1gt_3,962). happensAt(walking(lb1gt_3),962). orientation(lb1gt_4,90,962). coords(lb1gt_4,310,112,962).  visible(lb1gt_4,962). happensAt(inactive(lb1gt_4),962).
orientation(lb1gt_3,103,963). coords(lb1gt_3,282,89,963).  visible(lb1gt_3,963). happensAt(walking(lb1gt_3),963). orientation(lb1gt_4,90,963). coords(lb1gt_4,310,112,963).  visible(lb1gt_4,963). happensAt(inactive(lb1gt_4),963).
orientation(lb1gt_3,111,964). coords(lb1gt_3,282,88,964).  visible(lb1gt_3,964). happensAt(walking(lb1gt_3),964). orientation(lb1gt_4,90,964). coords(lb1gt_4,310,112,964).  visible(lb1gt_4,964). happensAt(inactive(lb1gt_4),964).
orientation(lb1gt_3,117,965). coords(lb1gt_3,280,88,965).  visible(lb1gt_3,965). happensAt(walking(lb1gt_3),965). orientation(lb1gt_4,90,965). coords(lb1gt_4,310,112,965).  visible(lb1gt_4,965). happensAt(inactive(lb1gt_4),965).
orientation(lb1gt_3,123,966). coords(lb1gt_3,280,88,966).  visible(lb1gt_3,966). happensAt(walking(lb1gt_3),966). orientation(lb1gt_4,90,966). coords(lb1gt_4,310,112,966).  visible(lb1gt_4,966). happensAt(inactive(lb1gt_4),966).
orientation(lb1gt_3,125,967). coords(lb1gt_3,278,87,967).  visible(lb1gt_3,967). happensAt(walking(lb1gt_3),967). orientation(lb1gt_4,90,967). coords(lb1gt_4,310,112,967).  visible(lb1gt_4,967). happensAt(inactive(lb1gt_4),967).
orientation(lb1gt_3,125,968). coords(lb1gt_3,277,87,968).  visible(lb1gt_3,968). happensAt(walking(lb1gt_3),968). orientation(lb1gt_4,90,968). coords(lb1gt_4,310,112,968).  visible(lb1gt_4,968). happensAt(inactive(lb1gt_4),968).
orientation(lb1gt_3,125,969). coords(lb1gt_3,276,88,969).  visible(lb1gt_3,969). happensAt(walking(lb1gt_3),969). orientation(lb1gt_4,90,969). coords(lb1gt_4,310,112,969).  visible(lb1gt_4,969). happensAt(inactive(lb1gt_4),969).
orientation(lb1gt_3,125,970). coords(lb1gt_3,272,88,970).  visible(lb1gt_3,970). happensAt(walking(lb1gt_3),970). orientation(lb1gt_4,90,970). coords(lb1gt_4,310,112,970).  visible(lb1gt_4,970). happensAt(inactive(lb1gt_4),970).
orientation(lb1gt_3,125,971). coords(lb1gt_3,269,88,971).  visible(lb1gt_3,971). happensAt(walking(lb1gt_3),971). orientation(lb1gt_4,90,971). coords(lb1gt_4,310,112,971).  visible(lb1gt_4,971). happensAt(inactive(lb1gt_4),971).
orientation(lb1gt_3,125,972). coords(lb1gt_3,266,89,972).  visible(lb1gt_3,972). happensAt(walking(lb1gt_3),972). orientation(lb1gt_4,90,972). coords(lb1gt_4,310,112,972).  visible(lb1gt_4,972). happensAt(inactive(lb1gt_4),972).
orientation(lb1gt_3,125,973). coords(lb1gt_3,263,88,973).  visible(lb1gt_3,973). happensAt(walking(lb1gt_3),973). orientation(lb1gt_4,90,973). coords(lb1gt_4,310,112,973).  visible(lb1gt_4,973). happensAt(inactive(lb1gt_4),973).
orientation(lb1gt_3,125,974). coords(lb1gt_3,262,88,974).  visible(lb1gt_3,974). happensAt(walking(lb1gt_3),974). orientation(lb1gt_4,90,974). coords(lb1gt_4,310,112,974).  visible(lb1gt_4,974). happensAt(inactive(lb1gt_4),974).
orientation(lb1gt_3,125,975). coords(lb1gt_3,261,88,975).  visible(lb1gt_3,975). happensAt(walking(lb1gt_3),975). orientation(lb1gt_4,90,975). coords(lb1gt_4,310,112,975).  visible(lb1gt_4,975). happensAt(inactive(lb1gt_4),975).
orientation(lb1gt_3,130,976). coords(lb1gt_3,260,87,976).  visible(lb1gt_3,976). happensAt(walking(lb1gt_3),976). orientation(lb1gt_4,90,976). coords(lb1gt_4,310,112,976).  visible(lb1gt_4,976). happensAt(inactive(lb1gt_4),976).
orientation(lb1gt_3,130,977). coords(lb1gt_3,258,86,977).  visible(lb1gt_3,977). happensAt(walking(lb1gt_3),977). orientation(lb1gt_4,90,977). coords(lb1gt_4,310,112,977).  visible(lb1gt_4,977). happensAt(inactive(lb1gt_4),977).
orientation(lb1gt_3,130,978). coords(lb1gt_3,257,86,978).  visible(lb1gt_3,978). happensAt(walking(lb1gt_3),978). orientation(lb1gt_4,90,978). coords(lb1gt_4,310,112,978).  visible(lb1gt_4,978). happensAt(inactive(lb1gt_4),978).
orientation(lb1gt_3,130,979). coords(lb1gt_3,256,86,979).  visible(lb1gt_3,979). happensAt(walking(lb1gt_3),979). orientation(lb1gt_4,90,979). coords(lb1gt_4,310,112,979).  visible(lb1gt_4,979). happensAt(inactive(lb1gt_4),979).
orientation(lb1gt_3,130,980). coords(lb1gt_3,254,86,980).  visible(lb1gt_3,980). happensAt(walking(lb1gt_3),980). orientation(lb1gt_4,90,980). coords(lb1gt_4,310,112,980).  visible(lb1gt_4,980). happensAt(inactive(lb1gt_4),980).
orientation(lb1gt_3,130,981). coords(lb1gt_3,251,86,981).  visible(lb1gt_3,981). happensAt(walking(lb1gt_3),981). orientation(lb1gt_4,90,981). coords(lb1gt_4,310,112,981).  visible(lb1gt_4,981). happensAt(inactive(lb1gt_4),981).
#show holdsAt/2.