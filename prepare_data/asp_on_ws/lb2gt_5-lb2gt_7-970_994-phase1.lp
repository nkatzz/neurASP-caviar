
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

	orientation(lb2gt_5,152,970). coords(lb2gt_5,102,127,970).  visible(lb2gt_5,970). happensAt(walking(lb2gt_5),970). orientation(lb2gt_7,0,970). coords(lb2gt_7,106,262,970).  visible(lb2gt_7,970). happensAt(walking(lb2gt_7),970).
orientation(lb2gt_5,152,971). coords(lb2gt_5,102,127,971).  visible(lb2gt_5,971). happensAt(walking(lb2gt_5),971). orientation(lb2gt_7,0,971). coords(lb2gt_7,105,262,971).  visible(lb2gt_7,971). happensAt(walking(lb2gt_7),971).
orientation(lb2gt_5,152,972). coords(lb2gt_5,102,127,972).  visible(lb2gt_5,972). happensAt(walking(lb2gt_5),972). orientation(lb2gt_7,0,972). coords(lb2gt_7,104,262,972).  visible(lb2gt_7,972). happensAt(walking(lb2gt_7),972).
orientation(lb2gt_5,152,973). coords(lb2gt_5,103,128,973).  visible(lb2gt_5,973). happensAt(walking(lb2gt_5),973). orientation(lb2gt_7,0,973). coords(lb2gt_7,104,262,973).  visible(lb2gt_7,973). happensAt(walking(lb2gt_7),973).
orientation(lb2gt_5,152,974). coords(lb2gt_5,104,129,974).  visible(lb2gt_5,974). happensAt(walking(lb2gt_5),974). orientation(lb2gt_7,0,974). coords(lb2gt_7,103,262,974).  visible(lb2gt_7,974). happensAt(walking(lb2gt_7),974).
orientation(lb2gt_5,152,975). coords(lb2gt_5,106,130,975).  visible(lb2gt_5,975). happensAt(walking(lb2gt_5),975). orientation(lb2gt_7,0,975). coords(lb2gt_7,102,263,975).  visible(lb2gt_7,975). happensAt(walking(lb2gt_7),975).
orientation(lb2gt_5,152,976). coords(lb2gt_5,107,131,976).  visible(lb2gt_5,976). happensAt(walking(lb2gt_5),976). orientation(lb2gt_7,0,976). coords(lb2gt_7,102,263,976).  visible(lb2gt_7,976). happensAt(walking(lb2gt_7),976).
orientation(lb2gt_5,152,977). coords(lb2gt_5,108,131,977).  visible(lb2gt_5,977). happensAt(walking(lb2gt_5),977). orientation(lb2gt_7,0,977). coords(lb2gt_7,101,263,977).  visible(lb2gt_7,977). happensAt(walking(lb2gt_7),977).
orientation(lb2gt_5,152,978). coords(lb2gt_5,109,131,978).  visible(lb2gt_5,978). happensAt(walking(lb2gt_5),978). orientation(lb2gt_7,0,978). coords(lb2gt_7,100,263,978).  visible(lb2gt_7,978). happensAt(walking(lb2gt_7),978).
orientation(lb2gt_5,152,979). coords(lb2gt_5,110,132,979).  visible(lb2gt_5,979). happensAt(walking(lb2gt_5),979). orientation(lb2gt_7,0,979). coords(lb2gt_7,100,263,979).  visible(lb2gt_7,979). happensAt(walking(lb2gt_7),979).
orientation(lb2gt_5,152,980). coords(lb2gt_5,110,132,980).  visible(lb2gt_5,980). happensAt(walking(lb2gt_5),980). orientation(lb2gt_7,0,980). coords(lb2gt_7,100,263,980).  visible(lb2gt_7,980). happensAt(walking(lb2gt_7),980).
orientation(lb2gt_5,152,981). coords(lb2gt_5,110,133,981).  visible(lb2gt_5,981). happensAt(walking(lb2gt_5),981). orientation(lb2gt_7,0,981). coords(lb2gt_7,100,263,981).  visible(lb2gt_7,981). happensAt(walking(lb2gt_7),981).
orientation(lb2gt_5,152,982). coords(lb2gt_5,110,133,982).  visible(lb2gt_5,982). happensAt(walking(lb2gt_5),982). orientation(lb2gt_7,0,982). coords(lb2gt_7,100,263,982).  visible(lb2gt_7,982). happensAt(walking(lb2gt_7),982).
orientation(lb2gt_5,152,983). coords(lb2gt_5,111,133,983).  visible(lb2gt_5,983). happensAt(walking(lb2gt_5),983). orientation(lb2gt_7,0,983). coords(lb2gt_7,100,263,983).  visible(lb2gt_7,983). happensAt(walking(lb2gt_7),983).
orientation(lb2gt_5,152,984). coords(lb2gt_5,111,133,984).  visible(lb2gt_5,984). happensAt(walking(lb2gt_5),984). orientation(lb2gt_7,0,984). coords(lb2gt_7,99,263,984).  visible(lb2gt_7,984). happensAt(walking(lb2gt_7),984).
orientation(lb2gt_5,152,985). coords(lb2gt_5,111,133,985).  visible(lb2gt_5,985). happensAt(walking(lb2gt_5),985). orientation(lb2gt_7,0,985). coords(lb2gt_7,99,263,985).  visible(lb2gt_7,985). happensAt(walking(lb2gt_7),985).
orientation(lb2gt_5,152,986). coords(lb2gt_5,112,133,986).  visible(lb2gt_5,986). happensAt(walking(lb2gt_5),986). orientation(lb2gt_7,0,986). coords(lb2gt_7,99,263,986).  visible(lb2gt_7,986). happensAt(walking(lb2gt_7),986).
orientation(lb2gt_5,152,987). coords(lb2gt_5,114,133,987).  visible(lb2gt_5,987). happensAt(walking(lb2gt_5),987). orientation(lb2gt_7,0,987). coords(lb2gt_7,99,263,987).  visible(lb2gt_7,987). happensAt(walking(lb2gt_7),987).
orientation(lb2gt_5,152,988). coords(lb2gt_5,116,135,988).  visible(lb2gt_5,988). happensAt(walking(lb2gt_5),988). orientation(lb2gt_7,0,988). coords(lb2gt_7,99,263,988).  visible(lb2gt_7,988). happensAt(walking(lb2gt_7),988).
orientation(lb2gt_5,152,989). coords(lb2gt_5,118,135,989).  visible(lb2gt_5,989). happensAt(walking(lb2gt_5),989). orientation(lb2gt_7,0,989). coords(lb2gt_7,99,263,989).  visible(lb2gt_7,989). happensAt(walking(lb2gt_7),989).
orientation(lb2gt_5,152,990). coords(lb2gt_5,119,135,990).  visible(lb2gt_5,990). happensAt(walking(lb2gt_5),990). orientation(lb2gt_7,0,990). coords(lb2gt_7,99,263,990).  visible(lb2gt_7,990). happensAt(walking(lb2gt_7),990).
orientation(lb2gt_5,152,991). coords(lb2gt_5,120,135,991).  visible(lb2gt_5,991). happensAt(walking(lb2gt_5),991). orientation(lb2gt_7,0,991). coords(lb2gt_7,99,263,991).  visible(lb2gt_7,991). happensAt(walking(lb2gt_7),991).
orientation(lb2gt_5,152,992). coords(lb2gt_5,122,136,992).  visible(lb2gt_5,992). happensAt(walking(lb2gt_5),992). orientation(lb2gt_7,0,992). coords(lb2gt_7,99,263,992).  visible(lb2gt_7,992). happensAt(walking(lb2gt_7),992).
orientation(lb2gt_5,152,993). coords(lb2gt_5,123,136,993).  visible(lb2gt_5,993). happensAt(walking(lb2gt_5),993). orientation(lb2gt_7,0,993). coords(lb2gt_7,99,263,993).  visible(lb2gt_7,993). happensAt(walking(lb2gt_7),993).
orientation(lb2gt_5,152,994). coords(lb2gt_5,124,137,994).  visible(lb2gt_5,994). happensAt(walking(lb2gt_5),994). orientation(lb2gt_7,0,994). coords(lb2gt_7,99,263,994).  visible(lb2gt_7,994). happensAt(walking(lb2gt_7),994).
#show holdsAt/2.