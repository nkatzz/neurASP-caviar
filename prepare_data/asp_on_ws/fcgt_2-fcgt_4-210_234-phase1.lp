
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

	orientation(fcgt_2,157,210). coords(fcgt_2,43,187,210).  visible(fcgt_2,210). happensAt(walking(fcgt_2),210). orientation(fcgt_4,172,210). coords(fcgt_4,84,205,210).  visible(fcgt_4,210). happensAt(walking(fcgt_4),210).
orientation(fcgt_2,157,211). coords(fcgt_2,43,185,211).  visible(fcgt_2,211). happensAt(walking(fcgt_2),211). orientation(fcgt_4,172,211). coords(fcgt_4,84,205,211).  visible(fcgt_4,211). happensAt(walking(fcgt_4),211).
orientation(fcgt_2,157,212). coords(fcgt_2,43,184,212).  visible(fcgt_2,212). happensAt(walking(fcgt_2),212). orientation(fcgt_4,172,212). coords(fcgt_4,84,205,212).  visible(fcgt_4,212). happensAt(walking(fcgt_4),212).
orientation(fcgt_2,157,213). coords(fcgt_2,44,184,213).  visible(fcgt_2,213). happensAt(walking(fcgt_2),213). orientation(fcgt_4,172,213). coords(fcgt_4,83,205,213).  visible(fcgt_4,213). happensAt(walking(fcgt_4),213).
orientation(fcgt_2,157,214). coords(fcgt_2,45,184,214).  visible(fcgt_2,214). happensAt(walking(fcgt_2),214). orientation(fcgt_4,172,214). coords(fcgt_4,83,205,214).  visible(fcgt_4,214). happensAt(walking(fcgt_4),214).
orientation(fcgt_2,157,215). coords(fcgt_2,46,184,215).  visible(fcgt_2,215). happensAt(walking(fcgt_2),215). orientation(fcgt_4,172,215). coords(fcgt_4,82,205,215).  visible(fcgt_4,215). happensAt(walking(fcgt_4),215).
orientation(fcgt_2,157,216). coords(fcgt_2,47,184,216).  visible(fcgt_2,216). happensAt(walking(fcgt_2),216). orientation(fcgt_4,172,216). coords(fcgt_4,82,205,216).  visible(fcgt_4,216). happensAt(walking(fcgt_4),216).
orientation(fcgt_2,157,217). coords(fcgt_2,47,183,217).  visible(fcgt_2,217). happensAt(walking(fcgt_2),217). orientation(fcgt_4,172,217). coords(fcgt_4,82,205,217).  visible(fcgt_4,217). happensAt(walking(fcgt_4),217).
orientation(fcgt_2,157,218). coords(fcgt_2,47,182,218).  visible(fcgt_2,218). happensAt(walking(fcgt_2),218). orientation(fcgt_4,172,218). coords(fcgt_4,82,205,218).  visible(fcgt_4,218). happensAt(walking(fcgt_4),218).
orientation(fcgt_2,157,219). coords(fcgt_2,47,181,219).  visible(fcgt_2,219). happensAt(walking(fcgt_2),219). orientation(fcgt_4,172,219). coords(fcgt_4,81,205,219).  visible(fcgt_4,219). happensAt(walking(fcgt_4),219).
orientation(fcgt_2,157,220). coords(fcgt_2,48,180,220).  visible(fcgt_2,220). happensAt(walking(fcgt_2),220). orientation(fcgt_4,172,220). coords(fcgt_4,81,205,220).  visible(fcgt_4,220). happensAt(walking(fcgt_4),220).
orientation(fcgt_2,157,221). coords(fcgt_2,48,179,221).  visible(fcgt_2,221). happensAt(walking(fcgt_2),221). orientation(fcgt_4,172,221). coords(fcgt_4,81,205,221).  visible(fcgt_4,221). happensAt(walking(fcgt_4),221).
orientation(fcgt_2,157,222). coords(fcgt_2,48,178,222).  visible(fcgt_2,222). happensAt(walking(fcgt_2),222). orientation(fcgt_4,172,222). coords(fcgt_4,81,205,222).  visible(fcgt_4,222). happensAt(walking(fcgt_4),222).
orientation(fcgt_2,157,223). coords(fcgt_2,48,177,223).  visible(fcgt_2,223). happensAt(walking(fcgt_2),223). orientation(fcgt_4,166,223). coords(fcgt_4,81,205,223).  visible(fcgt_4,223). happensAt(inactive(fcgt_4),223).
orientation(fcgt_2,157,224). coords(fcgt_2,48,176,224).  visible(fcgt_2,224). happensAt(walking(fcgt_2),224). orientation(fcgt_4,166,224). coords(fcgt_4,81,205,224).  visible(fcgt_4,224). happensAt(inactive(fcgt_4),224).
orientation(fcgt_2,157,225). coords(fcgt_2,49,176,225).  visible(fcgt_2,225). happensAt(walking(fcgt_2),225). orientation(fcgt_4,166,225). coords(fcgt_4,80,205,225).  visible(fcgt_4,225). happensAt(inactive(fcgt_4),225).
orientation(fcgt_2,157,226). coords(fcgt_2,50,176,226).  visible(fcgt_2,226). happensAt(walking(fcgt_2),226). orientation(fcgt_4,166,226). coords(fcgt_4,80,205,226).  visible(fcgt_4,226). happensAt(inactive(fcgt_4),226).
orientation(fcgt_2,157,227). coords(fcgt_2,51,175,227).  visible(fcgt_2,227). happensAt(walking(fcgt_2),227). orientation(fcgt_4,166,227). coords(fcgt_4,80,205,227).  visible(fcgt_4,227). happensAt(inactive(fcgt_4),227).
orientation(fcgt_2,157,228). coords(fcgt_2,51,175,228).  visible(fcgt_2,228). happensAt(walking(fcgt_2),228). orientation(fcgt_4,166,228). coords(fcgt_4,80,205,228).  visible(fcgt_4,228). happensAt(inactive(fcgt_4),228).
orientation(fcgt_2,157,229). coords(fcgt_2,51,175,229).  visible(fcgt_2,229). happensAt(walking(fcgt_2),229). orientation(fcgt_4,166,229). coords(fcgt_4,80,205,229).  visible(fcgt_4,229). happensAt(inactive(fcgt_4),229).
orientation(fcgt_2,157,230). coords(fcgt_2,52,174,230).  visible(fcgt_2,230). happensAt(walking(fcgt_2),230). orientation(fcgt_4,166,230). coords(fcgt_4,80,205,230).  visible(fcgt_4,230). happensAt(inactive(fcgt_4),230).
orientation(fcgt_2,157,231). coords(fcgt_2,53,174,231).  visible(fcgt_2,231). happensAt(walking(fcgt_2),231). orientation(fcgt_4,166,231). coords(fcgt_4,80,205,231).  visible(fcgt_4,231). happensAt(inactive(fcgt_4),231).
orientation(fcgt_2,157,232). coords(fcgt_2,53,172,232).  visible(fcgt_2,232). happensAt(walking(fcgt_2),232). orientation(fcgt_4,166,232). coords(fcgt_4,80,205,232).  visible(fcgt_4,232). happensAt(inactive(fcgt_4),232).
orientation(fcgt_2,157,233). coords(fcgt_2,53,171,233).  visible(fcgt_2,233). happensAt(walking(fcgt_2),233). orientation(fcgt_4,166,233). coords(fcgt_4,80,205,233).  visible(fcgt_4,233). happensAt(inactive(fcgt_4),233).
orientation(fcgt_2,157,234). coords(fcgt_2,53,170,234).  visible(fcgt_2,234). happensAt(walking(fcgt_2),234). orientation(fcgt_4,166,234). coords(fcgt_4,80,205,234).  visible(fcgt_4,234). happensAt(inactive(fcgt_4),234).
#show holdsAt/2.