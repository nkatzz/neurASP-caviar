
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

	orientation(fcgt_1,2,208). coords(fcgt_1,229,244,208).  visible(fcgt_1,208). happensAt(walking(fcgt_1),208). orientation(fcgt_2,166,208). coords(fcgt_2,42,188,208).  visible(fcgt_2,208). happensAt(walking(fcgt_2),208).
orientation(fcgt_1,2,209). coords(fcgt_1,228,242,209).  visible(fcgt_1,209). happensAt(walking(fcgt_1),209). orientation(fcgt_2,158,209). coords(fcgt_2,42,188,209).  visible(fcgt_2,209). happensAt(walking(fcgt_2),209).
orientation(fcgt_1,2,210). coords(fcgt_1,227,240,210).  visible(fcgt_1,210). happensAt(walking(fcgt_1),210). orientation(fcgt_2,157,210). coords(fcgt_2,43,187,210).  visible(fcgt_2,210). happensAt(walking(fcgt_2),210).
orientation(fcgt_1,177,211). coords(fcgt_1,226,240,211).  visible(fcgt_1,211). happensAt(walking(fcgt_1),211). orientation(fcgt_2,157,211). coords(fcgt_2,43,185,211).  visible(fcgt_2,211). happensAt(walking(fcgt_2),211).
orientation(fcgt_1,177,212). coords(fcgt_1,224,238,212).  visible(fcgt_1,212). happensAt(walking(fcgt_1),212). orientation(fcgt_2,157,212). coords(fcgt_2,43,184,212).  visible(fcgt_2,212). happensAt(walking(fcgt_2),212).
orientation(fcgt_1,169,213). coords(fcgt_1,222,237,213).  visible(fcgt_1,213). happensAt(walking(fcgt_1),213). orientation(fcgt_2,157,213). coords(fcgt_2,44,184,213).  visible(fcgt_2,213). happensAt(walking(fcgt_2),213).
orientation(fcgt_1,161,214). coords(fcgt_1,219,235,214).  visible(fcgt_1,214). happensAt(walking(fcgt_1),214). orientation(fcgt_2,157,214). coords(fcgt_2,45,184,214).  visible(fcgt_2,214). happensAt(walking(fcgt_2),214).
orientation(fcgt_1,161,215). coords(fcgt_1,215,232,215).  visible(fcgt_1,215). happensAt(walking(fcgt_1),215). orientation(fcgt_2,157,215). coords(fcgt_2,46,184,215).  visible(fcgt_2,215). happensAt(walking(fcgt_2),215).
orientation(fcgt_1,161,216). coords(fcgt_1,214,227,216).  visible(fcgt_1,216). happensAt(walking(fcgt_1),216). orientation(fcgt_2,157,216). coords(fcgt_2,47,184,216).  visible(fcgt_2,216). happensAt(walking(fcgt_2),216).
orientation(fcgt_1,171,217). coords(fcgt_1,213,222,217).  visible(fcgt_1,217). happensAt(walking(fcgt_1),217). orientation(fcgt_2,157,217). coords(fcgt_2,47,183,217).  visible(fcgt_2,217). happensAt(walking(fcgt_2),217).
orientation(fcgt_1,171,218). coords(fcgt_1,212,219,218).  visible(fcgt_1,218). happensAt(walking(fcgt_1),218). orientation(fcgt_2,157,218). coords(fcgt_2,47,182,218).  visible(fcgt_2,218). happensAt(walking(fcgt_2),218).
orientation(fcgt_1,171,219). coords(fcgt_1,211,215,219).  visible(fcgt_1,219). happensAt(walking(fcgt_1),219). orientation(fcgt_2,157,219). coords(fcgt_2,47,181,219).  visible(fcgt_2,219). happensAt(walking(fcgt_2),219).
orientation(fcgt_1,171,220). coords(fcgt_1,210,212,220).  visible(fcgt_1,220). happensAt(walking(fcgt_1),220). orientation(fcgt_2,157,220). coords(fcgt_2,48,180,220).  visible(fcgt_2,220). happensAt(walking(fcgt_2),220).
orientation(fcgt_1,171,221). coords(fcgt_1,210,210,221).  visible(fcgt_1,221). happensAt(walking(fcgt_1),221). orientation(fcgt_2,157,221). coords(fcgt_2,48,179,221).  visible(fcgt_2,221). happensAt(walking(fcgt_2),221).
orientation(fcgt_1,171,222). coords(fcgt_1,209,209,222).  visible(fcgt_1,222). happensAt(walking(fcgt_1),222). orientation(fcgt_2,157,222). coords(fcgt_2,48,178,222).  visible(fcgt_2,222). happensAt(walking(fcgt_2),222).
orientation(fcgt_1,171,223). coords(fcgt_1,206,206,223).  visible(fcgt_1,223). happensAt(walking(fcgt_1),223). orientation(fcgt_2,157,223). coords(fcgt_2,48,177,223).  visible(fcgt_2,223). happensAt(walking(fcgt_2),223).
orientation(fcgt_1,171,224). coords(fcgt_1,205,204,224).  visible(fcgt_1,224). happensAt(walking(fcgt_1),224). orientation(fcgt_2,157,224). coords(fcgt_2,48,176,224).  visible(fcgt_2,224). happensAt(walking(fcgt_2),224).
orientation(fcgt_1,171,225). coords(fcgt_1,203,202,225).  visible(fcgt_1,225). happensAt(walking(fcgt_1),225). orientation(fcgt_2,157,225). coords(fcgt_2,49,176,225).  visible(fcgt_2,225). happensAt(walking(fcgt_2),225).
orientation(fcgt_1,171,226). coords(fcgt_1,200,199,226).  visible(fcgt_1,226). happensAt(walking(fcgt_1),226). orientation(fcgt_2,157,226). coords(fcgt_2,50,176,226).  visible(fcgt_2,226). happensAt(walking(fcgt_2),226).
orientation(fcgt_1,171,227). coords(fcgt_1,197,197,227).  visible(fcgt_1,227). happensAt(walking(fcgt_1),227). orientation(fcgt_2,157,227). coords(fcgt_2,51,175,227).  visible(fcgt_2,227). happensAt(walking(fcgt_2),227).
orientation(fcgt_1,171,228). coords(fcgt_1,194,194,228).  visible(fcgt_1,228). happensAt(walking(fcgt_1),228). orientation(fcgt_2,157,228). coords(fcgt_2,51,175,228).  visible(fcgt_2,228). happensAt(walking(fcgt_2),228).
orientation(fcgt_1,167,229). coords(fcgt_1,191,194,229).  visible(fcgt_1,229). happensAt(walking(fcgt_1),229). orientation(fcgt_2,157,229). coords(fcgt_2,51,175,229).  visible(fcgt_2,229). happensAt(walking(fcgt_2),229).
orientation(fcgt_1,163,230). coords(fcgt_1,190,193,230).  visible(fcgt_1,230). happensAt(walking(fcgt_1),230). orientation(fcgt_2,157,230). coords(fcgt_2,52,174,230).  visible(fcgt_2,230). happensAt(walking(fcgt_2),230).
orientation(fcgt_1,163,231). coords(fcgt_1,189,192,231).  visible(fcgt_1,231). happensAt(walking(fcgt_1),231). orientation(fcgt_2,157,231). coords(fcgt_2,53,174,231).  visible(fcgt_2,231). happensAt(walking(fcgt_2),231).
orientation(fcgt_1,163,232). coords(fcgt_1,189,190,232).  visible(fcgt_1,232). happensAt(walking(fcgt_1),232). orientation(fcgt_2,157,232). coords(fcgt_2,53,172,232).  visible(fcgt_2,232). happensAt(walking(fcgt_2),232).
#show holdsAt/2.