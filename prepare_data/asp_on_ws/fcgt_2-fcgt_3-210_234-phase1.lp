
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

	orientation(fcgt_2,157,210). coords(fcgt_2,43,187,210).  visible(fcgt_2,210). happensAt(walking(fcgt_2),210). orientation(fcgt_3,149,210). coords(fcgt_3,100,166,210).  visible(fcgt_3,210). happensAt(walking(fcgt_3),210).
orientation(fcgt_2,157,211). coords(fcgt_2,43,185,211).  visible(fcgt_2,211). happensAt(walking(fcgt_2),211). orientation(fcgt_3,149,211). coords(fcgt_3,100,167,211).  visible(fcgt_3,211). happensAt(walking(fcgt_3),211).
orientation(fcgt_2,157,212). coords(fcgt_2,43,184,212).  visible(fcgt_2,212). happensAt(walking(fcgt_2),212). orientation(fcgt_3,149,212). coords(fcgt_3,100,167,212).  visible(fcgt_3,212). happensAt(walking(fcgt_3),212).
orientation(fcgt_2,157,213). coords(fcgt_2,44,184,213).  visible(fcgt_2,213). happensAt(walking(fcgt_2),213). orientation(fcgt_3,149,213). coords(fcgt_3,100,168,213).  visible(fcgt_3,213). happensAt(walking(fcgt_3),213).
orientation(fcgt_2,157,214). coords(fcgt_2,45,184,214).  visible(fcgt_2,214). happensAt(walking(fcgt_2),214). orientation(fcgt_3,149,214). coords(fcgt_3,100,168,214).  visible(fcgt_3,214). happensAt(walking(fcgt_3),214).
orientation(fcgt_2,157,215). coords(fcgt_2,46,184,215).  visible(fcgt_2,215). happensAt(walking(fcgt_2),215). orientation(fcgt_3,149,215). coords(fcgt_3,100,168,215).  visible(fcgt_3,215). happensAt(walking(fcgt_3),215).
orientation(fcgt_2,157,216). coords(fcgt_2,47,184,216).  visible(fcgt_2,216). happensAt(walking(fcgt_2),216). orientation(fcgt_3,149,216). coords(fcgt_3,100,168,216).  visible(fcgt_3,216). happensAt(walking(fcgt_3),216).
orientation(fcgt_2,157,217). coords(fcgt_2,47,183,217).  visible(fcgt_2,217). happensAt(walking(fcgt_2),217). orientation(fcgt_3,149,217). coords(fcgt_3,100,169,217).  visible(fcgt_3,217). happensAt(walking(fcgt_3),217).
orientation(fcgt_2,157,218). coords(fcgt_2,47,182,218).  visible(fcgt_2,218). happensAt(walking(fcgt_2),218). orientation(fcgt_3,149,218). coords(fcgt_3,100,169,218).  visible(fcgt_3,218). happensAt(walking(fcgt_3),218).
orientation(fcgt_2,157,219). coords(fcgt_2,47,181,219).  visible(fcgt_2,219). happensAt(walking(fcgt_2),219). orientation(fcgt_3,149,219). coords(fcgt_3,100,169,219).  visible(fcgt_3,219). happensAt(walking(fcgt_3),219).
orientation(fcgt_2,157,220). coords(fcgt_2,48,180,220).  visible(fcgt_2,220). happensAt(walking(fcgt_2),220). orientation(fcgt_3,149,220). coords(fcgt_3,100,170,220).  visible(fcgt_3,220). happensAt(walking(fcgt_3),220).
orientation(fcgt_2,157,221). coords(fcgt_2,48,179,221).  visible(fcgt_2,221). happensAt(walking(fcgt_2),221). orientation(fcgt_3,149,221). coords(fcgt_3,100,170,221).  visible(fcgt_3,221). happensAt(walking(fcgt_3),221).
orientation(fcgt_2,157,222). coords(fcgt_2,48,178,222).  visible(fcgt_2,222). happensAt(walking(fcgt_2),222). orientation(fcgt_3,149,222). coords(fcgt_3,100,170,222).  visible(fcgt_3,222). happensAt(walking(fcgt_3),222).
orientation(fcgt_2,157,223). coords(fcgt_2,48,177,223).  visible(fcgt_2,223). happensAt(walking(fcgt_2),223). orientation(fcgt_3,149,223). coords(fcgt_3,100,170,223).  visible(fcgt_3,223). happensAt(walking(fcgt_3),223).
orientation(fcgt_2,157,224). coords(fcgt_2,48,176,224).  visible(fcgt_2,224). happensAt(walking(fcgt_2),224). orientation(fcgt_3,149,224). coords(fcgt_3,100,170,224).  visible(fcgt_3,224). happensAt(walking(fcgt_3),224).
orientation(fcgt_2,157,225). coords(fcgt_2,49,176,225).  visible(fcgt_2,225). happensAt(walking(fcgt_2),225). orientation(fcgt_3,149,225). coords(fcgt_3,100,171,225).  visible(fcgt_3,225). happensAt(walking(fcgt_3),225).
orientation(fcgt_2,157,226). coords(fcgt_2,50,176,226).  visible(fcgt_2,226). happensAt(walking(fcgt_2),226). orientation(fcgt_3,149,226). coords(fcgt_3,100,171,226).  visible(fcgt_3,226). happensAt(walking(fcgt_3),226).
orientation(fcgt_2,157,227). coords(fcgt_2,51,175,227).  visible(fcgt_2,227). happensAt(walking(fcgt_2),227). orientation(fcgt_3,157,227). coords(fcgt_3,100,171,227).  visible(fcgt_3,227). happensAt(walking(fcgt_3),227).
orientation(fcgt_2,157,228). coords(fcgt_2,51,175,228).  visible(fcgt_2,228). happensAt(walking(fcgt_2),228). orientation(fcgt_3,159,228). coords(fcgt_3,100,171,228).  visible(fcgt_3,228). happensAt(walking(fcgt_3),228).
orientation(fcgt_2,157,229). coords(fcgt_2,51,175,229).  visible(fcgt_2,229). happensAt(walking(fcgt_2),229). orientation(fcgt_3,159,229). coords(fcgt_3,100,172,229).  visible(fcgt_3,229). happensAt(walking(fcgt_3),229).
orientation(fcgt_2,157,230). coords(fcgt_2,52,174,230).  visible(fcgt_2,230). happensAt(walking(fcgt_2),230). orientation(fcgt_3,159,230). coords(fcgt_3,100,172,230).  visible(fcgt_3,230). happensAt(walking(fcgt_3),230).
orientation(fcgt_2,157,231). coords(fcgt_2,53,174,231).  visible(fcgt_2,231). happensAt(walking(fcgt_2),231). orientation(fcgt_3,159,231). coords(fcgt_3,100,172,231).  visible(fcgt_3,231). happensAt(walking(fcgt_3),231).
orientation(fcgt_2,157,232). coords(fcgt_2,53,172,232).  visible(fcgt_2,232). happensAt(walking(fcgt_2),232). orientation(fcgt_3,159,232). coords(fcgt_3,100,172,232).  visible(fcgt_3,232). happensAt(walking(fcgt_3),232).
orientation(fcgt_2,157,233). coords(fcgt_2,53,171,233).  visible(fcgt_2,233). happensAt(walking(fcgt_2),233). orientation(fcgt_3,159,233). coords(fcgt_3,100,172,233).  visible(fcgt_3,233). happensAt(walking(fcgt_3),233).
orientation(fcgt_2,157,234). coords(fcgt_2,53,170,234).  visible(fcgt_2,234). happensAt(walking(fcgt_2),234). orientation(fcgt_3,159,234). coords(fcgt_3,100,172,234).  visible(fcgt_3,234). happensAt(walking(fcgt_3),234).
#show holdsAt/2.