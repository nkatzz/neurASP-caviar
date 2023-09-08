
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

	orientation(br1gt_1,170,216). coords(br1gt_1,30,159,216).  visible(br1gt_1,216). happensAt(walking(br1gt_1),216). orientation(br1gt_2,171,216). coords(br1gt_2,86,207,216).  visible(br1gt_2,216). happensAt(inactive(br1gt_2),216).
orientation(br1gt_1,170,217). coords(br1gt_1,30,159,217).  visible(br1gt_1,217). happensAt(walking(br1gt_1),217). orientation(br1gt_2,171,217). coords(br1gt_2,86,207,217).  visible(br1gt_2,217). happensAt(inactive(br1gt_2),217).
orientation(br1gt_1,170,218). coords(br1gt_1,30,159,218).  visible(br1gt_1,218). happensAt(walking(br1gt_1),218). orientation(br1gt_2,171,218). coords(br1gt_2,86,207,218).  visible(br1gt_2,218). happensAt(inactive(br1gt_2),218).
orientation(br1gt_1,170,219). coords(br1gt_1,30,159,219).  visible(br1gt_1,219). happensAt(walking(br1gt_1),219). orientation(br1gt_2,171,219). coords(br1gt_2,86,207,219).  visible(br1gt_2,219). happensAt(inactive(br1gt_2),219).
orientation(br1gt_1,170,220). coords(br1gt_1,29,160,220).  visible(br1gt_1,220). happensAt(walking(br1gt_1),220). orientation(br1gt_2,171,220). coords(br1gt_2,86,207,220).  visible(br1gt_2,220). happensAt(inactive(br1gt_2),220).
orientation(br1gt_1,170,221). coords(br1gt_1,29,161,221).  visible(br1gt_1,221). happensAt(walking(br1gt_1),221). orientation(br1gt_2,171,221). coords(br1gt_2,86,207,221).  visible(br1gt_2,221). happensAt(inactive(br1gt_2),221).
orientation(br1gt_1,170,222). coords(br1gt_1,28,160,222).  visible(br1gt_1,222). happensAt(walking(br1gt_1),222). orientation(br1gt_2,171,222). coords(br1gt_2,86,207,222).  visible(br1gt_2,222). happensAt(inactive(br1gt_2),222).
orientation(br1gt_1,170,223). coords(br1gt_1,28,161,223).  visible(br1gt_1,223). happensAt(walking(br1gt_1),223). orientation(br1gt_2,171,223). coords(br1gt_2,86,207,223).  visible(br1gt_2,223). happensAt(inactive(br1gt_2),223).
orientation(br1gt_1,170,224). coords(br1gt_1,26,161,224).  visible(br1gt_1,224). happensAt(walking(br1gt_1),224). orientation(br1gt_2,171,224). coords(br1gt_2,86,207,224).  visible(br1gt_2,224). happensAt(inactive(br1gt_2),224).
orientation(br1gt_1,170,225). coords(br1gt_1,27,163,225).  visible(br1gt_1,225). happensAt(walking(br1gt_1),225). orientation(br1gt_2,171,225). coords(br1gt_2,86,207,225).  visible(br1gt_2,225). happensAt(inactive(br1gt_2),225).
orientation(br1gt_1,170,226). coords(br1gt_1,27,162,226).  visible(br1gt_1,226). happensAt(walking(br1gt_1),226). orientation(br1gt_2,171,226). coords(br1gt_2,86,207,226).  visible(br1gt_2,226). happensAt(inactive(br1gt_2),226).
orientation(br1gt_1,170,227). coords(br1gt_1,26,163,227).  visible(br1gt_1,227). happensAt(walking(br1gt_1),227). orientation(br1gt_2,171,227). coords(br1gt_2,86,207,227).  visible(br1gt_2,227). happensAt(inactive(br1gt_2),227).
orientation(br1gt_1,170,228). coords(br1gt_1,27,162,228).  visible(br1gt_1,228). happensAt(walking(br1gt_1),228). orientation(br1gt_2,171,228). coords(br1gt_2,86,207,228).  visible(br1gt_2,228). happensAt(inactive(br1gt_2),228).
orientation(br1gt_1,170,229). coords(br1gt_1,26,163,229).  visible(br1gt_1,229). happensAt(walking(br1gt_1),229). orientation(br1gt_2,171,229). coords(br1gt_2,86,207,229).  visible(br1gt_2,229). happensAt(inactive(br1gt_2),229).
orientation(br1gt_1,170,230). coords(br1gt_1,27,163,230).  visible(br1gt_1,230). happensAt(walking(br1gt_1),230). orientation(br1gt_2,171,230). coords(br1gt_2,86,207,230).  visible(br1gt_2,230). happensAt(inactive(br1gt_2),230).
orientation(br1gt_1,170,231). coords(br1gt_1,27,164,231).  visible(br1gt_1,231). happensAt(walking(br1gt_1),231). orientation(br1gt_2,171,231). coords(br1gt_2,86,207,231).  visible(br1gt_2,231). happensAt(inactive(br1gt_2),231).
orientation(br1gt_1,170,232). coords(br1gt_1,26,165,232).  visible(br1gt_1,232). happensAt(walking(br1gt_1),232). orientation(br1gt_2,171,232). coords(br1gt_2,86,207,232).  visible(br1gt_2,232). happensAt(inactive(br1gt_2),232).
orientation(br1gt_1,170,233). coords(br1gt_1,27,165,233).  visible(br1gt_1,233). happensAt(walking(br1gt_1),233). orientation(br1gt_2,171,233). coords(br1gt_2,86,207,233).  visible(br1gt_2,233). happensAt(inactive(br1gt_2),233).
orientation(br1gt_1,170,234). coords(br1gt_1,27,165,234).  visible(br1gt_1,234). happensAt(walking(br1gt_1),234). orientation(br1gt_2,171,234). coords(br1gt_2,86,207,234).  visible(br1gt_2,234). happensAt(inactive(br1gt_2),234).
orientation(br1gt_1,170,235). coords(br1gt_1,26,165,235).  visible(br1gt_1,235). happensAt(walking(br1gt_1),235). orientation(br1gt_2,171,235). coords(br1gt_2,86,207,235).  visible(br1gt_2,235). happensAt(inactive(br1gt_2),235).
orientation(br1gt_1,170,236). coords(br1gt_1,27,166,236).  visible(br1gt_1,236). happensAt(walking(br1gt_1),236). orientation(br1gt_2,171,236). coords(br1gt_2,86,207,236).  visible(br1gt_2,236). happensAt(inactive(br1gt_2),236).
orientation(br1gt_1,170,237). coords(br1gt_1,27,166,237).  visible(br1gt_1,237). happensAt(walking(br1gt_1),237). orientation(br1gt_2,171,237). coords(br1gt_2,86,207,237).  visible(br1gt_2,237). happensAt(inactive(br1gt_2),237).
orientation(br1gt_1,170,238). coords(br1gt_1,26,166,238).  visible(br1gt_1,238). happensAt(walking(br1gt_1),238). orientation(br1gt_2,171,238). coords(br1gt_2,86,207,238).  visible(br1gt_2,238). happensAt(inactive(br1gt_2),238).
orientation(br1gt_1,166,239). coords(br1gt_1,27,168,239).  visible(br1gt_1,239). happensAt(walking(br1gt_1),239). orientation(br1gt_2,171,239). coords(br1gt_2,86,207,239).  visible(br1gt_2,239). happensAt(inactive(br1gt_2),239).
orientation(br1gt_1,166,240). coords(br1gt_1,26,168,240).  visible(br1gt_1,240). happensAt(walking(br1gt_1),240). orientation(br1gt_2,171,240). coords(br1gt_2,86,207,240).  visible(br1gt_2,240). happensAt(inactive(br1gt_2),240).
#show holdsAt/2.