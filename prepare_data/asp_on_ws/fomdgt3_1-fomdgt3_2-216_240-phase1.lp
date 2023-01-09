
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

	orientation(fomdgt3_1,150,216). coords(fomdgt3_1,73,96,216).  visible(fomdgt3_1,216). happensAt(walking(fomdgt3_1),216). orientation(fomdgt3_2,130,216). coords(fomdgt3_2,228,128,216).  visible(fomdgt3_2,216). happensAt(walking(fomdgt3_2),216).
orientation(fomdgt3_1,150,217). coords(fomdgt3_1,73,96,217).  visible(fomdgt3_1,217). happensAt(walking(fomdgt3_1),217). orientation(fomdgt3_2,130,217). coords(fomdgt3_2,228,128,217).  visible(fomdgt3_2,217). happensAt(walking(fomdgt3_2),217).
orientation(fomdgt3_1,150,218). coords(fomdgt3_1,73,96,218).  visible(fomdgt3_1,218). happensAt(walking(fomdgt3_1),218). orientation(fomdgt3_2,130,218). coords(fomdgt3_2,228,128,218).  visible(fomdgt3_2,218). happensAt(walking(fomdgt3_2),218).
orientation(fomdgt3_1,150,219). coords(fomdgt3_1,73,96,219).  visible(fomdgt3_1,219). happensAt(walking(fomdgt3_1),219). orientation(fomdgt3_2,130,219). coords(fomdgt3_2,228,128,219).  visible(fomdgt3_2,219). happensAt(walking(fomdgt3_2),219).
orientation(fomdgt3_1,150,220). coords(fomdgt3_1,73,96,220).  visible(fomdgt3_1,220). happensAt(walking(fomdgt3_1),220). orientation(fomdgt3_2,130,220). coords(fomdgt3_2,228,128,220).  visible(fomdgt3_2,220). happensAt(walking(fomdgt3_2),220).
orientation(fomdgt3_1,150,221). coords(fomdgt3_1,69,94,221).  visible(fomdgt3_1,221). happensAt(walking(fomdgt3_1),221). orientation(fomdgt3_2,130,221). coords(fomdgt3_2,228,128,221).  visible(fomdgt3_2,221). happensAt(walking(fomdgt3_2),221).
orientation(fomdgt3_1,150,222). coords(fomdgt3_1,69,94,222).  visible(fomdgt3_1,222). happensAt(walking(fomdgt3_1),222). orientation(fomdgt3_2,130,222). coords(fomdgt3_2,228,128,222).  visible(fomdgt3_2,222). happensAt(walking(fomdgt3_2),222).
orientation(fomdgt3_1,150,223). coords(fomdgt3_1,69,94,223).  visible(fomdgt3_1,223). happensAt(walking(fomdgt3_1),223). orientation(fomdgt3_2,130,223). coords(fomdgt3_2,228,128,223).  visible(fomdgt3_2,223). happensAt(walking(fomdgt3_2),223).
orientation(fomdgt3_1,150,224). coords(fomdgt3_1,69,94,224).  visible(fomdgt3_1,224). happensAt(walking(fomdgt3_1),224). orientation(fomdgt3_2,130,224). coords(fomdgt3_2,228,128,224).  visible(fomdgt3_2,224). happensAt(walking(fomdgt3_2),224).
orientation(fomdgt3_1,150,225). coords(fomdgt3_1,69,94,225).  visible(fomdgt3_1,225). happensAt(walking(fomdgt3_1),225). orientation(fomdgt3_2,130,225). coords(fomdgt3_2,228,128,225).  visible(fomdgt3_2,225). happensAt(walking(fomdgt3_2),225).
orientation(fomdgt3_1,150,226). coords(fomdgt3_1,68,93,226).  visible(fomdgt3_1,226). happensAt(walking(fomdgt3_1),226). orientation(fomdgt3_2,130,226). coords(fomdgt3_2,228,128,226).  visible(fomdgt3_2,226). happensAt(walking(fomdgt3_2),226).
orientation(fomdgt3_1,150,227). coords(fomdgt3_1,68,93,227).  visible(fomdgt3_1,227). happensAt(walking(fomdgt3_1),227). orientation(fomdgt3_2,130,227). coords(fomdgt3_2,228,128,227).  visible(fomdgt3_2,227). happensAt(walking(fomdgt3_2),227).
orientation(fomdgt3_1,150,228). coords(fomdgt3_1,68,93,228).  visible(fomdgt3_1,228). happensAt(walking(fomdgt3_1),228). orientation(fomdgt3_2,130,228). coords(fomdgt3_2,228,128,228).  visible(fomdgt3_2,228). happensAt(walking(fomdgt3_2),228).
orientation(fomdgt3_1,150,229). coords(fomdgt3_1,68,93,229).  visible(fomdgt3_1,229). happensAt(walking(fomdgt3_1),229). orientation(fomdgt3_2,130,229). coords(fomdgt3_2,228,128,229).  visible(fomdgt3_2,229). happensAt(walking(fomdgt3_2),229).
orientation(fomdgt3_1,150,230). coords(fomdgt3_1,68,93,230).  visible(fomdgt3_1,230). happensAt(walking(fomdgt3_1),230). orientation(fomdgt3_2,130,230). coords(fomdgt3_2,228,128,230).  visible(fomdgt3_2,230). happensAt(walking(fomdgt3_2),230).
orientation(fomdgt3_1,150,231). coords(fomdgt3_1,68,93,231).  visible(fomdgt3_1,231). happensAt(walking(fomdgt3_1),231). orientation(fomdgt3_2,130,231). coords(fomdgt3_2,228,128,231).  visible(fomdgt3_2,231). happensAt(walking(fomdgt3_2),231).
orientation(fomdgt3_1,150,232). coords(fomdgt3_1,68,93,232).  visible(fomdgt3_1,232). happensAt(walking(fomdgt3_1),232). orientation(fomdgt3_2,130,232). coords(fomdgt3_2,228,128,232).  visible(fomdgt3_2,232). happensAt(walking(fomdgt3_2),232).
orientation(fomdgt3_1,150,233). coords(fomdgt3_1,68,93,233).  visible(fomdgt3_1,233). happensAt(walking(fomdgt3_1),233). orientation(fomdgt3_2,130,233). coords(fomdgt3_2,228,128,233).  visible(fomdgt3_2,233). happensAt(walking(fomdgt3_2),233).
orientation(fomdgt3_1,150,234). coords(fomdgt3_1,68,93,234).  visible(fomdgt3_1,234). happensAt(walking(fomdgt3_1),234). orientation(fomdgt3_2,130,234). coords(fomdgt3_2,228,128,234).  visible(fomdgt3_2,234). happensAt(walking(fomdgt3_2),234).
orientation(fomdgt3_1,150,235). coords(fomdgt3_1,66,93,235).  visible(fomdgt3_1,235). happensAt(walking(fomdgt3_1),235). orientation(fomdgt3_2,130,235). coords(fomdgt3_2,228,128,235).  visible(fomdgt3_2,235). happensAt(walking(fomdgt3_2),235).
orientation(fomdgt3_1,150,236). coords(fomdgt3_1,66,93,236).  visible(fomdgt3_1,236). happensAt(walking(fomdgt3_1),236). orientation(fomdgt3_2,130,236). coords(fomdgt3_2,228,128,236).  visible(fomdgt3_2,236). happensAt(walking(fomdgt3_2),236).
orientation(fomdgt3_1,150,237). coords(fomdgt3_1,66,93,237).  visible(fomdgt3_1,237). happensAt(walking(fomdgt3_1),237). orientation(fomdgt3_2,130,237). coords(fomdgt3_2,228,128,237).  visible(fomdgt3_2,237). happensAt(walking(fomdgt3_2),237).
orientation(fomdgt3_1,150,238). coords(fomdgt3_1,66,93,238).  visible(fomdgt3_1,238). happensAt(walking(fomdgt3_1),238). orientation(fomdgt3_2,130,238). coords(fomdgt3_2,228,128,238).  visible(fomdgt3_2,238). happensAt(walking(fomdgt3_2),238).
orientation(fomdgt3_1,150,239). coords(fomdgt3_1,66,93,239).  visible(fomdgt3_1,239). happensAt(walking(fomdgt3_1),239). orientation(fomdgt3_2,130,239). coords(fomdgt3_2,228,128,239).  visible(fomdgt3_2,239). happensAt(walking(fomdgt3_2),239).
orientation(fomdgt3_1,150,240). coords(fomdgt3_1,66,93,240).  visible(fomdgt3_1,240). happensAt(walking(fomdgt3_1),240). orientation(fomdgt3_2,130,240). coords(fomdgt3_2,228,128,240).  visible(fomdgt3_2,240). happensAt(walking(fomdgt3_2),240).
#show holdsAt/2.