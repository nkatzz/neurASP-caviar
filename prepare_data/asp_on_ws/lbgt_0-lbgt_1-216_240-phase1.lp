
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

	orientation(lbgt_0,147,216). coords(lbgt_0,175,152,216).  visible(lbgt_0,216). happensAt(walking(lbgt_0),216). orientation(lbgt_1,147,216). coords(lbgt_1,81,81,216).  visible(lbgt_1,216). happensAt(walking(lbgt_1),216).
orientation(lbgt_0,150,217). coords(lbgt_0,174,148,217).  visible(lbgt_0,217). happensAt(walking(lbgt_0),217). orientation(lbgt_1,147,217). coords(lbgt_1,81,80,217).  visible(lbgt_1,217). happensAt(walking(lbgt_1),217).
orientation(lbgt_0,150,218). coords(lbgt_0,173,146,218).  visible(lbgt_0,218). happensAt(walking(lbgt_0),218). orientation(lbgt_1,147,218). coords(lbgt_1,81,79,218).  visible(lbgt_1,218). happensAt(walking(lbgt_1),218).
orientation(lbgt_0,150,219). coords(lbgt_0,173,144,219).  visible(lbgt_0,219). happensAt(walking(lbgt_0),219). orientation(lbgt_1,147,219). coords(lbgt_1,81,79,219).  visible(lbgt_1,219). happensAt(walking(lbgt_1),219).
orientation(lbgt_0,155,220). coords(lbgt_0,173,144,220).  visible(lbgt_0,220). happensAt(walking(lbgt_0),220). orientation(lbgt_1,147,220). coords(lbgt_1,81,77,220).  visible(lbgt_1,220). happensAt(walking(lbgt_1),220).
orientation(lbgt_0,155,221). coords(lbgt_0,172,144,221).  visible(lbgt_0,221). happensAt(walking(lbgt_0),221). orientation(lbgt_1,147,221). coords(lbgt_1,81,76,221).  visible(lbgt_1,221). happensAt(walking(lbgt_1),221).
orientation(lbgt_0,155,222). coords(lbgt_0,172,144,222).  visible(lbgt_0,222). happensAt(walking(lbgt_0),222). orientation(lbgt_1,147,222). coords(lbgt_1,81,76,222).  visible(lbgt_1,222). happensAt(walking(lbgt_1),222).
orientation(lbgt_0,155,223). coords(lbgt_0,172,144,223).  visible(lbgt_0,223). happensAt(walking(lbgt_0),223). orientation(lbgt_1,147,223). coords(lbgt_1,81,75,223).  visible(lbgt_1,223). happensAt(walking(lbgt_1),223).
orientation(lbgt_0,155,224). coords(lbgt_0,171,144,224).  visible(lbgt_0,224). happensAt(walking(lbgt_0),224). orientation(lbgt_1,147,224). coords(lbgt_1,81,74,224).  visible(lbgt_1,224). happensAt(walking(lbgt_1),224).
orientation(lbgt_0,155,225). coords(lbgt_0,170,143,225).  visible(lbgt_0,225). happensAt(walking(lbgt_0),225). orientation(lbgt_1,147,225). coords(lbgt_1,81,74,225).  visible(lbgt_1,225). happensAt(walking(lbgt_1),225).
orientation(lbgt_0,155,226). coords(lbgt_0,169,142,226).  visible(lbgt_0,226). happensAt(walking(lbgt_0),226). orientation(lbgt_1,147,226). coords(lbgt_1,81,73,226).  visible(lbgt_1,226). happensAt(walking(lbgt_1),226).
orientation(lbgt_0,155,227). coords(lbgt_0,168,140,227).  visible(lbgt_0,227). happensAt(walking(lbgt_0),227). orientation(lbgt_1,147,227). coords(lbgt_1,81,73,227).  visible(lbgt_1,227). happensAt(walking(lbgt_1),227).
orientation(lbgt_0,158,228). coords(lbgt_0,167,139,228).  visible(lbgt_0,228). happensAt(walking(lbgt_0),228). orientation(lbgt_1,147,228). coords(lbgt_1,81,72,228).  visible(lbgt_1,228). happensAt(walking(lbgt_1),228).
orientation(lbgt_0,158,229). coords(lbgt_0,166,137,229).  visible(lbgt_0,229). happensAt(walking(lbgt_0),229). orientation(lbgt_1,147,229). coords(lbgt_1,81,72,229).  visible(lbgt_1,229). happensAt(walking(lbgt_1),229).
orientation(lbgt_0,158,230). coords(lbgt_0,163,137,230).  visible(lbgt_0,230). happensAt(walking(lbgt_0),230). orientation(lbgt_1,147,230). coords(lbgt_1,81,71,230).  visible(lbgt_1,230). happensAt(walking(lbgt_1),230).
orientation(lbgt_0,158,231). coords(lbgt_0,162,136,231).  visible(lbgt_0,231). happensAt(walking(lbgt_0),231). orientation(lbgt_1,147,231). coords(lbgt_1,81,70,231).  visible(lbgt_1,231). happensAt(walking(lbgt_1),231).
orientation(lbgt_0,158,232). coords(lbgt_0,160,136,232).  visible(lbgt_0,232). happensAt(walking(lbgt_0),232). orientation(lbgt_1,147,232). coords(lbgt_1,81,69,232).  visible(lbgt_1,232). happensAt(walking(lbgt_1),232).
orientation(lbgt_0,158,233). coords(lbgt_0,159,135,233).  visible(lbgt_0,233). happensAt(walking(lbgt_0),233). orientation(lbgt_1,147,233). coords(lbgt_1,81,68,233).  visible(lbgt_1,233). happensAt(walking(lbgt_1),233).
orientation(lbgt_0,152,234). coords(lbgt_0,158,135,234).  visible(lbgt_0,234). happensAt(walking(lbgt_0),234). orientation(lbgt_1,147,234). coords(lbgt_1,81,68,234).  visible(lbgt_1,234). happensAt(walking(lbgt_1),234).
orientation(lbgt_0,152,235). coords(lbgt_0,158,135,235).  visible(lbgt_0,235). happensAt(walking(lbgt_0),235). orientation(lbgt_1,147,235). coords(lbgt_1,82,68,235).  visible(lbgt_1,235). happensAt(walking(lbgt_1),235).
orientation(lbgt_0,152,236). coords(lbgt_0,158,134,236).  visible(lbgt_0,236). happensAt(walking(lbgt_0),236). orientation(lbgt_1,147,236). coords(lbgt_1,82,68,236).  visible(lbgt_1,236). happensAt(walking(lbgt_1),236).
orientation(lbgt_0,152,237). coords(lbgt_0,158,134,237).  visible(lbgt_0,237). happensAt(walking(lbgt_0),237). orientation(lbgt_1,147,237). coords(lbgt_1,82,68,237).  visible(lbgt_1,237). happensAt(walking(lbgt_1),237).
orientation(lbgt_0,152,238). coords(lbgt_0,157,133,238).  visible(lbgt_0,238). happensAt(walking(lbgt_0),238). orientation(lbgt_1,147,238). coords(lbgt_1,82,68,238).  visible(lbgt_1,238). happensAt(walking(lbgt_1),238).
orientation(lbgt_0,152,239). coords(lbgt_0,157,133,239).  visible(lbgt_0,239). happensAt(walking(lbgt_0),239). orientation(lbgt_1,147,239). coords(lbgt_1,82,68,239).  visible(lbgt_1,239). happensAt(walking(lbgt_1),239).
orientation(lbgt_0,148,240). coords(lbgt_0,156,132,240).  visible(lbgt_0,240). happensAt(walking(lbgt_0),240). orientation(lbgt_1,147,240). coords(lbgt_1,82,68,240).  visible(lbgt_1,240). happensAt(walking(lbgt_1),240).
#show holdsAt/2.