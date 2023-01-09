
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

	orientation(ms3ggt_0,154,218). coords(ms3ggt_0,58,155,218).  visible(ms3ggt_0,218). happensAt(walking(ms3ggt_0),218). orientation(ms3ggt_1,159,218). coords(ms3ggt_1,44,165,218).  visible(ms3ggt_1,218). happensAt(walking(ms3ggt_1),218).
orientation(ms3ggt_0,154,219). coords(ms3ggt_0,58,155,219).  visible(ms3ggt_0,219). happensAt(walking(ms3ggt_0),219). orientation(ms3ggt_1,159,219). coords(ms3ggt_1,44,165,219).  visible(ms3ggt_1,219). happensAt(walking(ms3ggt_1),219).
orientation(ms3ggt_0,154,220). coords(ms3ggt_0,58,155,220).  visible(ms3ggt_0,220). happensAt(walking(ms3ggt_0),220). orientation(ms3ggt_1,159,220). coords(ms3ggt_1,44,165,220).  visible(ms3ggt_1,220). happensAt(walking(ms3ggt_1),220).
orientation(ms3ggt_0,154,221). coords(ms3ggt_0,58,155,221).  visible(ms3ggt_0,221). happensAt(inactive(ms3ggt_0),221). orientation(ms3ggt_1,159,221). coords(ms3ggt_1,45,165,221).  visible(ms3ggt_1,221). happensAt(walking(ms3ggt_1),221).
orientation(ms3ggt_0,154,222). coords(ms3ggt_0,58,155,222).  visible(ms3ggt_0,222). happensAt(inactive(ms3ggt_0),222). orientation(ms3ggt_1,159,222). coords(ms3ggt_1,45,165,222).  visible(ms3ggt_1,222). happensAt(walking(ms3ggt_1),222).
orientation(ms3ggt_0,154,223). coords(ms3ggt_0,58,155,223).  visible(ms3ggt_0,223). happensAt(inactive(ms3ggt_0),223). orientation(ms3ggt_1,159,223). coords(ms3ggt_1,45,165,223).  visible(ms3ggt_1,223). happensAt(walking(ms3ggt_1),223).
orientation(ms3ggt_0,154,224). coords(ms3ggt_0,58,155,224).  visible(ms3ggt_0,224). happensAt(inactive(ms3ggt_0),224). orientation(ms3ggt_1,159,224). coords(ms3ggt_1,46,165,224).  visible(ms3ggt_1,224). happensAt(walking(ms3ggt_1),224).
orientation(ms3ggt_0,154,225). coords(ms3ggt_0,58,155,225).  visible(ms3ggt_0,225). happensAt(inactive(ms3ggt_0),225). orientation(ms3ggt_1,159,225). coords(ms3ggt_1,45,164,225).  visible(ms3ggt_1,225). happensAt(walking(ms3ggt_1),225).
orientation(ms3ggt_0,154,226). coords(ms3ggt_0,58,155,226).  visible(ms3ggt_0,226). happensAt(inactive(ms3ggt_0),226). orientation(ms3ggt_1,159,226). coords(ms3ggt_1,45,164,226).  visible(ms3ggt_1,226). happensAt(walking(ms3ggt_1),226).
orientation(ms3ggt_0,154,227). coords(ms3ggt_0,58,155,227).  visible(ms3ggt_0,227). happensAt(inactive(ms3ggt_0),227). orientation(ms3ggt_1,159,227). coords(ms3ggt_1,46,164,227).  visible(ms3ggt_1,227). happensAt(walking(ms3ggt_1),227).
orientation(ms3ggt_0,154,228). coords(ms3ggt_0,58,155,228).  visible(ms3ggt_0,228). happensAt(inactive(ms3ggt_0),228). orientation(ms3ggt_1,159,228). coords(ms3ggt_1,46,164,228).  visible(ms3ggt_1,228). happensAt(walking(ms3ggt_1),228).
orientation(ms3ggt_0,154,229). coords(ms3ggt_0,58,155,229).  visible(ms3ggt_0,229). happensAt(inactive(ms3ggt_0),229). orientation(ms3ggt_1,159,229). coords(ms3ggt_1,46,164,229).  visible(ms3ggt_1,229). happensAt(walking(ms3ggt_1),229).
orientation(ms3ggt_0,154,230). coords(ms3ggt_0,58,155,230).  visible(ms3ggt_0,230). happensAt(inactive(ms3ggt_0),230). orientation(ms3ggt_1,163,230). coords(ms3ggt_1,46,164,230).  visible(ms3ggt_1,230). happensAt(walking(ms3ggt_1),230).
orientation(ms3ggt_0,154,231). coords(ms3ggt_0,58,155,231).  visible(ms3ggt_0,231). happensAt(inactive(ms3ggt_0),231). orientation(ms3ggt_1,163,231). coords(ms3ggt_1,46,164,231).  visible(ms3ggt_1,231). happensAt(walking(ms3ggt_1),231).
orientation(ms3ggt_0,154,232). coords(ms3ggt_0,58,155,232).  visible(ms3ggt_0,232). happensAt(inactive(ms3ggt_0),232). orientation(ms3ggt_1,163,232). coords(ms3ggt_1,47,163,232).  visible(ms3ggt_1,232). happensAt(walking(ms3ggt_1),232).
orientation(ms3ggt_0,154,233). coords(ms3ggt_0,58,155,233).  visible(ms3ggt_0,233). happensAt(inactive(ms3ggt_0),233). orientation(ms3ggt_1,163,233). coords(ms3ggt_1,48,164,233).  visible(ms3ggt_1,233). happensAt(walking(ms3ggt_1),233).
orientation(ms3ggt_0,154,234). coords(ms3ggt_0,58,155,234).  visible(ms3ggt_0,234). happensAt(inactive(ms3ggt_0),234). orientation(ms3ggt_1,163,234). coords(ms3ggt_1,48,164,234).  visible(ms3ggt_1,234). happensAt(walking(ms3ggt_1),234).
orientation(ms3ggt_0,154,235). coords(ms3ggt_0,58,155,235).  visible(ms3ggt_0,235). happensAt(inactive(ms3ggt_0),235). orientation(ms3ggt_1,167,235). coords(ms3ggt_1,50,164,235).  visible(ms3ggt_1,235). happensAt(walking(ms3ggt_1),235).
orientation(ms3ggt_0,154,236). coords(ms3ggt_0,58,155,236).  visible(ms3ggt_0,236). happensAt(inactive(ms3ggt_0),236). orientation(ms3ggt_1,167,236). coords(ms3ggt_1,50,164,236).  visible(ms3ggt_1,236). happensAt(walking(ms3ggt_1),236).
orientation(ms3ggt_0,154,237). coords(ms3ggt_0,58,155,237).  visible(ms3ggt_0,237). happensAt(inactive(ms3ggt_0),237). orientation(ms3ggt_1,167,237). coords(ms3ggt_1,50,164,237).  visible(ms3ggt_1,237). happensAt(walking(ms3ggt_1),237).
orientation(ms3ggt_0,154,238). coords(ms3ggt_0,58,155,238).  visible(ms3ggt_0,238). happensAt(inactive(ms3ggt_0),238). orientation(ms3ggt_1,167,238). coords(ms3ggt_1,51,164,238).  visible(ms3ggt_1,238). happensAt(walking(ms3ggt_1),238).
orientation(ms3ggt_0,154,239). coords(ms3ggt_0,58,155,239).  visible(ms3ggt_0,239). happensAt(inactive(ms3ggt_0),239). orientation(ms3ggt_1,167,239). coords(ms3ggt_1,52,164,239).  visible(ms3ggt_1,239). happensAt(walking(ms3ggt_1),239).
orientation(ms3ggt_0,154,240). coords(ms3ggt_0,58,155,240).  visible(ms3ggt_0,240). happensAt(inactive(ms3ggt_0),240). orientation(ms3ggt_1,167,240). coords(ms3ggt_1,52,165,240).  visible(ms3ggt_1,240). happensAt(walking(ms3ggt_1),240).
orientation(ms3ggt_0,154,241). coords(ms3ggt_0,58,155,241).  visible(ms3ggt_0,241). happensAt(inactive(ms3ggt_0),241). orientation(ms3ggt_1,167,241). coords(ms3ggt_1,52,165,241).  visible(ms3ggt_1,241). happensAt(walking(ms3ggt_1),241).
orientation(ms3ggt_0,154,242). coords(ms3ggt_0,58,155,242).  visible(ms3ggt_0,242). happensAt(inactive(ms3ggt_0),242). orientation(ms3ggt_1,167,242). coords(ms3ggt_1,52,165,242).  visible(ms3ggt_1,242). happensAt(walking(ms3ggt_1),242).
#show holdsAt/2.