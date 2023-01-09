
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

	orientation(ms3ggt_0,145,206). coords(ms3ggt_0,60,152,206).  visible(ms3ggt_0,206). happensAt(walking(ms3ggt_0),206). orientation(ms3ggt_1,154,206). coords(ms3ggt_1,41,165,206).  visible(ms3ggt_1,206). happensAt(walking(ms3ggt_1),206).
orientation(ms3ggt_0,145,207). coords(ms3ggt_0,60,152,207).  visible(ms3ggt_0,207). happensAt(walking(ms3ggt_0),207). orientation(ms3ggt_1,154,207). coords(ms3ggt_1,41,165,207).  visible(ms3ggt_1,207). happensAt(walking(ms3ggt_1),207).
orientation(ms3ggt_0,145,208). coords(ms3ggt_0,60,152,208).  visible(ms3ggt_0,208). happensAt(walking(ms3ggt_0),208). orientation(ms3ggt_1,154,208). coords(ms3ggt_1,41,165,208).  visible(ms3ggt_1,208). happensAt(walking(ms3ggt_1),208).
orientation(ms3ggt_0,145,209). coords(ms3ggt_0,60,153,209).  visible(ms3ggt_0,209). happensAt(walking(ms3ggt_0),209). orientation(ms3ggt_1,154,209). coords(ms3ggt_1,41,165,209).  visible(ms3ggt_1,209). happensAt(walking(ms3ggt_1),209).
orientation(ms3ggt_0,145,210). coords(ms3ggt_0,59,153,210).  visible(ms3ggt_0,210). happensAt(walking(ms3ggt_0),210). orientation(ms3ggt_1,154,210). coords(ms3ggt_1,42,165,210).  visible(ms3ggt_1,210). happensAt(walking(ms3ggt_1),210).
orientation(ms3ggt_0,145,211). coords(ms3ggt_0,59,154,211).  visible(ms3ggt_0,211). happensAt(walking(ms3ggt_0),211). orientation(ms3ggt_1,154,211). coords(ms3ggt_1,42,165,211).  visible(ms3ggt_1,211). happensAt(walking(ms3ggt_1),211).
orientation(ms3ggt_0,150,212). coords(ms3ggt_0,59,153,212).  visible(ms3ggt_0,212). happensAt(walking(ms3ggt_0),212). orientation(ms3ggt_1,159,212). coords(ms3ggt_1,42,164,212).  visible(ms3ggt_1,212). happensAt(walking(ms3ggt_1),212).
orientation(ms3ggt_0,154,213). coords(ms3ggt_0,58,153,213).  visible(ms3ggt_0,213). happensAt(walking(ms3ggt_0),213). orientation(ms3ggt_1,159,213). coords(ms3ggt_1,44,165,213).  visible(ms3ggt_1,213). happensAt(walking(ms3ggt_1),213).
orientation(ms3ggt_0,154,214). coords(ms3ggt_0,58,153,214).  visible(ms3ggt_0,214). happensAt(walking(ms3ggt_0),214). orientation(ms3ggt_1,159,214). coords(ms3ggt_1,44,165,214).  visible(ms3ggt_1,214). happensAt(walking(ms3ggt_1),214).
orientation(ms3ggt_0,154,215). coords(ms3ggt_0,58,153,215).  visible(ms3ggt_0,215). happensAt(walking(ms3ggt_0),215). orientation(ms3ggt_1,159,215). coords(ms3ggt_1,44,165,215).  visible(ms3ggt_1,215). happensAt(walking(ms3ggt_1),215).
orientation(ms3ggt_0,154,216). coords(ms3ggt_0,58,153,216).  visible(ms3ggt_0,216). happensAt(walking(ms3ggt_0),216). orientation(ms3ggt_1,159,216). coords(ms3ggt_1,44,165,216).  visible(ms3ggt_1,216). happensAt(walking(ms3ggt_1),216).
orientation(ms3ggt_0,154,217). coords(ms3ggt_0,58,154,217).  visible(ms3ggt_0,217). happensAt(walking(ms3ggt_0),217). orientation(ms3ggt_1,159,217). coords(ms3ggt_1,44,165,217).  visible(ms3ggt_1,217). happensAt(walking(ms3ggt_1),217).
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
#show holdsAt/2.