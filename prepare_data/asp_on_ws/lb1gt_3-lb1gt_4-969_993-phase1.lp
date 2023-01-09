
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
orientation(lb1gt_3,132,982). coords(lb1gt_3,248,86,982).  visible(lb1gt_3,982). happensAt(walking(lb1gt_3),982). orientation(lb1gt_4,90,982). coords(lb1gt_4,310,112,982).  visible(lb1gt_4,982). happensAt(inactive(lb1gt_4),982).
orientation(lb1gt_3,134,983). coords(lb1gt_3,244,86,983).  visible(lb1gt_3,983). happensAt(walking(lb1gt_3),983). orientation(lb1gt_4,90,983). coords(lb1gt_4,310,112,983).  visible(lb1gt_4,983). happensAt(inactive(lb1gt_4),983).
orientation(lb1gt_3,134,984). coords(lb1gt_3,240,86,984).  visible(lb1gt_3,984). happensAt(walking(lb1gt_3),984). orientation(lb1gt_4,90,984). coords(lb1gt_4,310,112,984).  visible(lb1gt_4,984). happensAt(inactive(lb1gt_4),984).
orientation(lb1gt_3,134,985). coords(lb1gt_3,236,85,985).  visible(lb1gt_3,985). happensAt(walking(lb1gt_3),985). orientation(lb1gt_4,90,985). coords(lb1gt_4,310,112,985).  visible(lb1gt_4,985). happensAt(inactive(lb1gt_4),985).
orientation(lb1gt_3,134,986). coords(lb1gt_3,233,85,986).  visible(lb1gt_3,986). happensAt(walking(lb1gt_3),986). orientation(lb1gt_4,90,986). coords(lb1gt_4,310,112,986).  visible(lb1gt_4,986). happensAt(inactive(lb1gt_4),986).
orientation(lb1gt_3,134,987). coords(lb1gt_3,231,85,987).  visible(lb1gt_3,987). happensAt(walking(lb1gt_3),987). orientation(lb1gt_4,90,987). coords(lb1gt_4,310,112,987).  visible(lb1gt_4,987). happensAt(inactive(lb1gt_4),987).
orientation(lb1gt_3,134,988). coords(lb1gt_3,230,85,988).  visible(lb1gt_3,988). happensAt(walking(lb1gt_3),988). orientation(lb1gt_4,90,988). coords(lb1gt_4,310,112,988).  visible(lb1gt_4,988). happensAt(inactive(lb1gt_4),988).
orientation(lb1gt_3,134,989). coords(lb1gt_3,229,85,989).  visible(lb1gt_3,989). happensAt(walking(lb1gt_3),989). orientation(lb1gt_4,90,989). coords(lb1gt_4,310,112,989).  visible(lb1gt_4,989). happensAt(inactive(lb1gt_4),989).
orientation(lb1gt_3,134,990). coords(lb1gt_3,227,85,990).  visible(lb1gt_3,990). happensAt(walking(lb1gt_3),990). orientation(lb1gt_4,90,990). coords(lb1gt_4,310,112,990).  visible(lb1gt_4,990). happensAt(inactive(lb1gt_4),990).
orientation(lb1gt_3,134,991). coords(lb1gt_3,226,85,991).  visible(lb1gt_3,991). happensAt(walking(lb1gt_3),991). orientation(lb1gt_4,90,991). coords(lb1gt_4,310,112,991).  visible(lb1gt_4,991). happensAt(inactive(lb1gt_4),991).
orientation(lb1gt_3,134,992). coords(lb1gt_3,225,85,992).  visible(lb1gt_3,992). happensAt(walking(lb1gt_3),992). orientation(lb1gt_4,90,992). coords(lb1gt_4,310,112,992).  visible(lb1gt_4,992). happensAt(inactive(lb1gt_4),992).
orientation(lb1gt_3,134,993). coords(lb1gt_3,223,85,993).  visible(lb1gt_3,993). happensAt(walking(lb1gt_3),993). orientation(lb1gt_4,90,993). coords(lb1gt_4,310,112,993).  visible(lb1gt_4,993). happensAt(inactive(lb1gt_4),993).
#show holdsAt/2.