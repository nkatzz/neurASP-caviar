
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
orientation(wk2gt_1,133,981). coords(wk2gt_1,130,43,981).  visible(wk2gt_1,981). happensAt(walking(wk2gt_1),981). orientation(wk2gt_5,150,981). coords(wk2gt_5,94,108,981).  visible(wk2gt_5,981). happensAt(walking(wk2gt_5),981).
orientation(wk2gt_1,133,982). coords(wk2gt_1,130,43,982).  visible(wk2gt_1,982). happensAt(walking(wk2gt_1),982). orientation(wk2gt_5,150,982). coords(wk2gt_5,95,108,982).  visible(wk2gt_5,982). happensAt(walking(wk2gt_5),982).
orientation(wk2gt_1,133,983). coords(wk2gt_1,130,43,983).  visible(wk2gt_1,983). happensAt(walking(wk2gt_1),983). orientation(wk2gt_5,150,983). coords(wk2gt_5,95,108,983).  visible(wk2gt_5,983). happensAt(walking(wk2gt_5),983).
orientation(wk2gt_1,133,984). coords(wk2gt_1,130,43,984).  visible(wk2gt_1,984). happensAt(walking(wk2gt_1),984). orientation(wk2gt_5,150,984). coords(wk2gt_5,95,109,984).  visible(wk2gt_5,984). happensAt(walking(wk2gt_5),984).
orientation(wk2gt_1,133,985). coords(wk2gt_1,130,43,985).  visible(wk2gt_1,985). happensAt(walking(wk2gt_1),985). orientation(wk2gt_5,150,985). coords(wk2gt_5,95,110,985).  visible(wk2gt_5,985). happensAt(walking(wk2gt_5),985).
orientation(wk2gt_1,133,986). coords(wk2gt_1,130,43,986).  visible(wk2gt_1,986). happensAt(walking(wk2gt_1),986). orientation(wk2gt_5,150,986). coords(wk2gt_5,95,110,986).  visible(wk2gt_5,986). happensAt(walking(wk2gt_5),986).
orientation(wk2gt_1,133,987). coords(wk2gt_1,130,42,987).  visible(wk2gt_1,987). happensAt(walking(wk2gt_1),987). orientation(wk2gt_5,150,987). coords(wk2gt_5,95,112,987).  visible(wk2gt_5,987). happensAt(walking(wk2gt_5),987).
orientation(wk2gt_1,133,988). coords(wk2gt_1,130,43,988).  visible(wk2gt_1,988). happensAt(walking(wk2gt_1),988). orientation(wk2gt_5,150,988). coords(wk2gt_5,97,113,988).  visible(wk2gt_5,988). happensAt(walking(wk2gt_5),988).
orientation(wk2gt_1,133,989). coords(wk2gt_1,131,42,989).  visible(wk2gt_1,989). happensAt(walking(wk2gt_1),989). orientation(wk2gt_5,150,989). coords(wk2gt_5,98,113,989).  visible(wk2gt_5,989). happensAt(walking(wk2gt_5),989).
orientation(wk2gt_1,133,990). coords(wk2gt_1,131,42,990).  visible(wk2gt_1,990). happensAt(walking(wk2gt_1),990). orientation(wk2gt_5,150,990). coords(wk2gt_5,99,114,990).  visible(wk2gt_5,990). happensAt(walking(wk2gt_5),990).
orientation(wk2gt_1,133,991). coords(wk2gt_1,132,42,991).  visible(wk2gt_1,991). happensAt(walking(wk2gt_1),991). orientation(wk2gt_5,150,991). coords(wk2gt_5,101,114,991).  visible(wk2gt_5,991). happensAt(walking(wk2gt_5),991).
orientation(wk2gt_1,133,992). coords(wk2gt_1,132,42,992).  visible(wk2gt_1,992). happensAt(walking(wk2gt_1),992). orientation(wk2gt_5,150,992). coords(wk2gt_5,101,114,992).  visible(wk2gt_5,992). happensAt(walking(wk2gt_5),992).
#show holdsAt/2.