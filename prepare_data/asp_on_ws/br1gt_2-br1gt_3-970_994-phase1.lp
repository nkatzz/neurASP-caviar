
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
orientation(br1gt_2,166,983). coords(br1gt_2,106,208,983).  visible(br1gt_2,983). happensAt(walking(br1gt_2),983). orientation(br1gt_3,122,983). coords(br1gt_3,290,94,983).  visible(br1gt_3,983). happensAt(walking(br1gt_3),983).
orientation(br1gt_2,166,984). coords(br1gt_2,106,208,984).  visible(br1gt_2,984). happensAt(walking(br1gt_2),984). orientation(br1gt_3,122,984). coords(br1gt_3,287,95,984).  visible(br1gt_3,984). happensAt(walking(br1gt_3),984).
orientation(br1gt_2,166,985). coords(br1gt_2,107,208,985).  visible(br1gt_2,985). happensAt(walking(br1gt_2),985). orientation(br1gt_3,122,985). coords(br1gt_3,284,95,985).  visible(br1gt_3,985). happensAt(walking(br1gt_3),985).
orientation(br1gt_2,166,986). coords(br1gt_2,107,208,986).  visible(br1gt_2,986). happensAt(walking(br1gt_2),986). orientation(br1gt_3,118,986). coords(br1gt_3,283,96,986).  visible(br1gt_3,986). happensAt(walking(br1gt_3),986).
orientation(br1gt_2,166,987). coords(br1gt_2,107,208,987).  visible(br1gt_2,987). happensAt(walking(br1gt_2),987). orientation(br1gt_3,118,987). coords(br1gt_3,282,96,987).  visible(br1gt_3,987). happensAt(walking(br1gt_3),987).
orientation(br1gt_2,166,988). coords(br1gt_2,107,208,988).  visible(br1gt_2,988). happensAt(walking(br1gt_2),988). orientation(br1gt_3,118,988). coords(br1gt_3,281,96,988).  visible(br1gt_3,988). happensAt(walking(br1gt_3),988).
orientation(br1gt_2,166,989). coords(br1gt_2,108,208,989).  visible(br1gt_2,989). happensAt(walking(br1gt_2),989). orientation(br1gt_3,118,989). coords(br1gt_3,281,96,989).  visible(br1gt_3,989). happensAt(walking(br1gt_3),989).
orientation(br1gt_2,166,990). coords(br1gt_2,108,208,990).  visible(br1gt_2,990). happensAt(walking(br1gt_2),990). orientation(br1gt_3,118,990). coords(br1gt_3,280,96,990).  visible(br1gt_3,990). happensAt(walking(br1gt_3),990).
orientation(br1gt_2,166,991). coords(br1gt_2,108,208,991).  visible(br1gt_2,991). happensAt(walking(br1gt_2),991). orientation(br1gt_3,118,991). coords(br1gt_3,279,95,991).  visible(br1gt_3,991). happensAt(walking(br1gt_3),991).
orientation(br1gt_2,174,992). coords(br1gt_2,108,208,992).  visible(br1gt_2,992). happensAt(walking(br1gt_2),992). orientation(br1gt_3,123,992). coords(br1gt_3,279,95,992).  visible(br1gt_3,992). happensAt(walking(br1gt_3),992).
orientation(br1gt_2,174,993). coords(br1gt_2,108,208,993).  visible(br1gt_2,993). happensAt(walking(br1gt_2),993). orientation(br1gt_3,123,993). coords(br1gt_3,279,95,993).  visible(br1gt_3,993). happensAt(walking(br1gt_3),993).
orientation(br1gt_2,174,994). coords(br1gt_2,108,209,994).  visible(br1gt_2,994). happensAt(walking(br1gt_2),994). orientation(br1gt_3,123,994). coords(br1gt_3,278,95,994).  visible(br1gt_3,994). happensAt(walking(br1gt_3),994).
#show holdsAt/2.