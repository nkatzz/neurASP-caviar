
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

	orientation(lbbcgt_5,154,969). coords(lbbcgt_5,187,142,969).  visible(lbbcgt_5,969). happensAt(walking(lbbcgt_5),969). orientation(lbbcgt_6,151,969). coords(lbbcgt_6,77,87,969).  visible(lbbcgt_6,969). happensAt(walking(lbbcgt_6),969).
orientation(lbbcgt_5,154,970). coords(lbbcgt_5,187,141,970).  visible(lbbcgt_5,970). happensAt(walking(lbbcgt_5),970). orientation(lbbcgt_6,151,970). coords(lbbcgt_6,76,87,970).  visible(lbbcgt_6,970). happensAt(walking(lbbcgt_6),970).
orientation(lbbcgt_5,154,971). coords(lbbcgt_5,188,139,971).  visible(lbbcgt_5,971). happensAt(walking(lbbcgt_5),971). orientation(lbbcgt_6,151,971). coords(lbbcgt_6,76,87,971).  visible(lbbcgt_6,971). happensAt(walking(lbbcgt_6),971).
orientation(lbbcgt_5,154,972). coords(lbbcgt_5,188,137,972).  visible(lbbcgt_5,972). happensAt(walking(lbbcgt_5),972). orientation(lbbcgt_6,151,972). coords(lbbcgt_6,75,88,972).  visible(lbbcgt_6,972). happensAt(walking(lbbcgt_6),972).
orientation(lbbcgt_5,154,973). coords(lbbcgt_5,188,135,973).  visible(lbbcgt_5,973). happensAt(walking(lbbcgt_5),973). orientation(lbbcgt_6,151,973). coords(lbbcgt_6,76,88,973).  visible(lbbcgt_6,973). happensAt(walking(lbbcgt_6),973).
orientation(lbbcgt_5,154,974). coords(lbbcgt_5,188,133,974).  visible(lbbcgt_5,974). happensAt(walking(lbbcgt_5),974). orientation(lbbcgt_6,151,974). coords(lbbcgt_6,75,88,974).  visible(lbbcgt_6,974). happensAt(walking(lbbcgt_6),974).
orientation(lbbcgt_5,154,975). coords(lbbcgt_5,188,132,975).  visible(lbbcgt_5,975). happensAt(walking(lbbcgt_5),975). orientation(lbbcgt_6,151,975). coords(lbbcgt_6,75,88,975).  visible(lbbcgt_6,975). happensAt(walking(lbbcgt_6),975).
orientation(lbbcgt_5,154,976). coords(lbbcgt_5,188,132,976).  visible(lbbcgt_5,976). happensAt(walking(lbbcgt_5),976). orientation(lbbcgt_6,151,976). coords(lbbcgt_6,75,88,976).  visible(lbbcgt_6,976). happensAt(walking(lbbcgt_6),976).
orientation(lbbcgt_5,154,977). coords(lbbcgt_5,188,131,977).  visible(lbbcgt_5,977). happensAt(walking(lbbcgt_5),977). orientation(lbbcgt_6,151,977). coords(lbbcgt_6,75,89,977).  visible(lbbcgt_6,977). happensAt(walking(lbbcgt_6),977).
orientation(lbbcgt_5,154,978). coords(lbbcgt_5,188,131,978).  visible(lbbcgt_5,978). happensAt(walking(lbbcgt_5),978). orientation(lbbcgt_6,151,978). coords(lbbcgt_6,75,89,978).  visible(lbbcgt_6,978). happensAt(walking(lbbcgt_6),978).
orientation(lbbcgt_5,154,979). coords(lbbcgt_5,188,130,979).  visible(lbbcgt_5,979). happensAt(walking(lbbcgt_5),979). orientation(lbbcgt_6,151,979). coords(lbbcgt_6,75,89,979).  visible(lbbcgt_6,979). happensAt(walking(lbbcgt_6),979).
orientation(lbbcgt_5,154,980). coords(lbbcgt_5,188,130,980).  visible(lbbcgt_5,980). happensAt(walking(lbbcgt_5),980). orientation(lbbcgt_6,151,980). coords(lbbcgt_6,74,89,980).  visible(lbbcgt_6,980). happensAt(walking(lbbcgt_6),980).
orientation(lbbcgt_5,154,981). coords(lbbcgt_5,188,129,981).  visible(lbbcgt_5,981). happensAt(walking(lbbcgt_5),981). orientation(lbbcgt_6,151,981). coords(lbbcgt_6,73,90,981).  visible(lbbcgt_6,981). happensAt(walking(lbbcgt_6),981).
orientation(lbbcgt_5,154,982). coords(lbbcgt_5,188,129,982).  visible(lbbcgt_5,982). happensAt(walking(lbbcgt_5),982). orientation(lbbcgt_6,151,982). coords(lbbcgt_6,72,90,982).  visible(lbbcgt_6,982). happensAt(walking(lbbcgt_6),982).
orientation(lbbcgt_5,154,983). coords(lbbcgt_5,188,128,983).  visible(lbbcgt_5,983). happensAt(walking(lbbcgt_5),983). orientation(lbbcgt_6,151,983). coords(lbbcgt_6,72,90,983).  visible(lbbcgt_6,983). happensAt(walking(lbbcgt_6),983).
orientation(lbbcgt_5,154,984). coords(lbbcgt_5,188,127,984).  visible(lbbcgt_5,984). happensAt(walking(lbbcgt_5),984). orientation(lbbcgt_6,151,984). coords(lbbcgt_6,71,90,984).  visible(lbbcgt_6,984). happensAt(walking(lbbcgt_6),984).
orientation(lbbcgt_5,154,985). coords(lbbcgt_5,187,126,985).  visible(lbbcgt_5,985). happensAt(walking(lbbcgt_5),985). orientation(lbbcgt_6,151,985). coords(lbbcgt_6,71,90,985).  visible(lbbcgt_6,985). happensAt(walking(lbbcgt_6),985).
orientation(lbbcgt_5,154,986). coords(lbbcgt_5,187,126,986).  visible(lbbcgt_5,986). happensAt(walking(lbbcgt_5),986). orientation(lbbcgt_6,151,986). coords(lbbcgt_6,70,90,986).  visible(lbbcgt_6,986). happensAt(walking(lbbcgt_6),986).
orientation(lbbcgt_5,154,987). coords(lbbcgt_5,186,125,987).  visible(lbbcgt_5,987). happensAt(walking(lbbcgt_5),987). orientation(lbbcgt_6,151,987). coords(lbbcgt_6,70,90,987).  visible(lbbcgt_6,987). happensAt(walking(lbbcgt_6),987).
orientation(lbbcgt_5,154,988). coords(lbbcgt_5,184,124,988).  visible(lbbcgt_5,988). happensAt(walking(lbbcgt_5),988). orientation(lbbcgt_6,151,988). coords(lbbcgt_6,70,90,988).  visible(lbbcgt_6,988). happensAt(walking(lbbcgt_6),988).
orientation(lbbcgt_5,154,989). coords(lbbcgt_5,183,122,989).  visible(lbbcgt_5,989). happensAt(walking(lbbcgt_5),989). orientation(lbbcgt_6,151,989). coords(lbbcgt_6,70,91,989).  visible(lbbcgt_6,989). happensAt(walking(lbbcgt_6),989).
orientation(lbbcgt_5,154,990). coords(lbbcgt_5,182,121,990).  visible(lbbcgt_5,990). happensAt(walking(lbbcgt_5),990). orientation(lbbcgt_6,151,990). coords(lbbcgt_6,70,91,990).  visible(lbbcgt_6,990). happensAt(walking(lbbcgt_6),990).
orientation(lbbcgt_5,154,991). coords(lbbcgt_5,181,119,991).  visible(lbbcgt_5,991). happensAt(active(lbbcgt_5),991). orientation(lbbcgt_6,151,991). coords(lbbcgt_6,69,91,991).  visible(lbbcgt_6,991). happensAt(walking(lbbcgt_6),991).
orientation(lbbcgt_5,154,992). coords(lbbcgt_5,181,118,992).  visible(lbbcgt_5,992). happensAt(active(lbbcgt_5),992). orientation(lbbcgt_6,151,992). coords(lbbcgt_6,69,91,992).  visible(lbbcgt_6,992). happensAt(walking(lbbcgt_6),992).
orientation(lbbcgt_5,154,993). coords(lbbcgt_5,180,118,993).  visible(lbbcgt_5,993). happensAt(active(lbbcgt_5),993). orientation(lbbcgt_6,151,993). coords(lbbcgt_6,69,91,993).  visible(lbbcgt_6,993). happensAt(walking(lbbcgt_6),993).
#show holdsAt/2.