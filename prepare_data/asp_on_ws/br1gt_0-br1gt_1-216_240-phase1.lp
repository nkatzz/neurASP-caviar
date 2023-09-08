
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

	orientation(br1gt_0,53,216). coords(br1gt_0,307,282,216).  visible(br1gt_0,216). happensAt(walking(br1gt_0),216). orientation(br1gt_1,170,216). coords(br1gt_1,30,159,216).  visible(br1gt_1,216). happensAt(walking(br1gt_1),216).
orientation(br1gt_0,53,217). coords(br1gt_0,307,282,217).  visible(br1gt_0,217). happensAt(walking(br1gt_0),217). orientation(br1gt_1,170,217). coords(br1gt_1,30,159,217).  visible(br1gt_1,217). happensAt(walking(br1gt_1),217).
orientation(br1gt_0,53,218). coords(br1gt_0,307,282,218).  visible(br1gt_0,218). happensAt(walking(br1gt_0),218). orientation(br1gt_1,170,218). coords(br1gt_1,30,159,218).  visible(br1gt_1,218). happensAt(walking(br1gt_1),218).
orientation(br1gt_0,53,219). coords(br1gt_0,307,282,219).  visible(br1gt_0,219). happensAt(walking(br1gt_0),219). orientation(br1gt_1,170,219). coords(br1gt_1,30,159,219).  visible(br1gt_1,219). happensAt(walking(br1gt_1),219).
orientation(br1gt_0,53,220). coords(br1gt_0,307,282,220).  visible(br1gt_0,220). happensAt(walking(br1gt_0),220). orientation(br1gt_1,170,220). coords(br1gt_1,29,160,220).  visible(br1gt_1,220). happensAt(walking(br1gt_1),220).
orientation(br1gt_0,53,221). coords(br1gt_0,307,282,221).  visible(br1gt_0,221). happensAt(walking(br1gt_0),221). orientation(br1gt_1,170,221). coords(br1gt_1,29,161,221).  visible(br1gt_1,221). happensAt(walking(br1gt_1),221).
orientation(br1gt_0,53,222). coords(br1gt_0,307,282,222).  visible(br1gt_0,222). happensAt(walking(br1gt_0),222). orientation(br1gt_1,170,222). coords(br1gt_1,28,160,222).  visible(br1gt_1,222). happensAt(walking(br1gt_1),222).
orientation(br1gt_0,53,223). coords(br1gt_0,307,282,223).  visible(br1gt_0,223). happensAt(walking(br1gt_0),223). orientation(br1gt_1,170,223). coords(br1gt_1,28,161,223).  visible(br1gt_1,223). happensAt(walking(br1gt_1),223).
orientation(br1gt_0,53,224). coords(br1gt_0,307,282,224).  visible(br1gt_0,224). happensAt(walking(br1gt_0),224). orientation(br1gt_1,170,224). coords(br1gt_1,26,161,224).  visible(br1gt_1,224). happensAt(walking(br1gt_1),224).
orientation(br1gt_0,53,225). coords(br1gt_0,309,282,225).  visible(br1gt_0,225). happensAt(walking(br1gt_0),225). orientation(br1gt_1,170,225). coords(br1gt_1,27,163,225).  visible(br1gt_1,225). happensAt(walking(br1gt_1),225).
orientation(br1gt_0,53,226). coords(br1gt_0,309,282,226).  visible(br1gt_0,226). happensAt(walking(br1gt_0),226). orientation(br1gt_1,170,226). coords(br1gt_1,27,162,226).  visible(br1gt_1,226). happensAt(walking(br1gt_1),226).
orientation(br1gt_0,53,227). coords(br1gt_0,309,282,227).  visible(br1gt_0,227). happensAt(walking(br1gt_0),227). orientation(br1gt_1,170,227). coords(br1gt_1,26,163,227).  visible(br1gt_1,227). happensAt(walking(br1gt_1),227).
orientation(br1gt_0,53,228). coords(br1gt_0,309,282,228).  visible(br1gt_0,228). happensAt(walking(br1gt_0),228). orientation(br1gt_1,170,228). coords(br1gt_1,27,162,228).  visible(br1gt_1,228). happensAt(walking(br1gt_1),228).
orientation(br1gt_0,53,229). coords(br1gt_0,309,282,229).  visible(br1gt_0,229). happensAt(walking(br1gt_0),229). orientation(br1gt_1,170,229). coords(br1gt_1,26,163,229).  visible(br1gt_1,229). happensAt(walking(br1gt_1),229).
orientation(br1gt_0,53,230). coords(br1gt_0,309,282,230).  visible(br1gt_0,230). happensAt(walking(br1gt_0),230). orientation(br1gt_1,170,230). coords(br1gt_1,27,163,230).  visible(br1gt_1,230). happensAt(walking(br1gt_1),230).
orientation(br1gt_0,53,231). coords(br1gt_0,309,282,231).  visible(br1gt_0,231). happensAt(walking(br1gt_0),231). orientation(br1gt_1,170,231). coords(br1gt_1,27,164,231).  visible(br1gt_1,231). happensAt(walking(br1gt_1),231).
orientation(br1gt_0,53,232). coords(br1gt_0,309,282,232).  visible(br1gt_0,232). happensAt(walking(br1gt_0),232). orientation(br1gt_1,170,232). coords(br1gt_1,26,165,232).  visible(br1gt_1,232). happensAt(walking(br1gt_1),232).
orientation(br1gt_0,53,233). coords(br1gt_0,309,282,233).  visible(br1gt_0,233). happensAt(walking(br1gt_0),233). orientation(br1gt_1,170,233). coords(br1gt_1,27,165,233).  visible(br1gt_1,233). happensAt(walking(br1gt_1),233).
orientation(br1gt_0,53,234). coords(br1gt_0,309,282,234).  visible(br1gt_0,234). happensAt(walking(br1gt_0),234). orientation(br1gt_1,170,234). coords(br1gt_1,27,165,234).  visible(br1gt_1,234). happensAt(walking(br1gt_1),234).
orientation(br1gt_0,53,235). coords(br1gt_0,311,282,235).  visible(br1gt_0,235). happensAt(walking(br1gt_0),235). orientation(br1gt_1,170,235). coords(br1gt_1,26,165,235).  visible(br1gt_1,235). happensAt(walking(br1gt_1),235).
orientation(br1gt_0,53,236). coords(br1gt_0,311,282,236).  visible(br1gt_0,236). happensAt(walking(br1gt_0),236). orientation(br1gt_1,170,236). coords(br1gt_1,27,166,236).  visible(br1gt_1,236). happensAt(walking(br1gt_1),236).
orientation(br1gt_0,53,237). coords(br1gt_0,314,283,237).  visible(br1gt_0,237). happensAt(walking(br1gt_0),237). orientation(br1gt_1,170,237). coords(br1gt_1,27,166,237).  visible(br1gt_1,237). happensAt(walking(br1gt_1),237).
orientation(br1gt_0,53,238). coords(br1gt_0,314,284,238).  visible(br1gt_0,238). happensAt(walking(br1gt_0),238). orientation(br1gt_1,170,238). coords(br1gt_1,26,166,238).  visible(br1gt_1,238). happensAt(walking(br1gt_1),238).
orientation(br1gt_0,53,239). coords(br1gt_0,314,285,239).  visible(br1gt_0,239). happensAt(walking(br1gt_0),239). orientation(br1gt_1,166,239). coords(br1gt_1,27,168,239).  visible(br1gt_1,239). happensAt(walking(br1gt_1),239).
orientation(br1gt_0,53,240). coords(br1gt_0,314,286,240).  happensAt(disappear(br1gt_0),240). happensAt(walking(br1gt_0),240). orientation(br1gt_1,166,240). coords(br1gt_1,26,168,240).  visible(br1gt_1,240). happensAt(walking(br1gt_1),240).
#show holdsAt/2.