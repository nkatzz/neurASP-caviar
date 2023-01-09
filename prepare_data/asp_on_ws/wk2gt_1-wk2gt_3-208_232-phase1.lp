
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

	orientation(wk2gt_1,129,208). coords(wk2gt_1,115,53,208).  visible(wk2gt_1,208). happensAt(inactive(wk2gt_1),208). orientation(wk2gt_3,38,208). coords(wk2gt_3,280,260,208).  visible(wk2gt_3,208). happensAt(walking(wk2gt_3),208).
orientation(wk2gt_1,129,209). coords(wk2gt_1,115,53,209).  visible(wk2gt_1,209). happensAt(inactive(wk2gt_1),209). orientation(wk2gt_3,38,209). coords(wk2gt_3,281,262,209).  visible(wk2gt_3,209). happensAt(walking(wk2gt_3),209).
orientation(wk2gt_1,129,210). coords(wk2gt_1,115,53,210).  visible(wk2gt_1,210). happensAt(inactive(wk2gt_1),210). orientation(wk2gt_3,38,210). coords(wk2gt_3,282,265,210).  visible(wk2gt_3,210). happensAt(walking(wk2gt_3),210).
orientation(wk2gt_1,129,211). coords(wk2gt_1,115,53,211).  visible(wk2gt_1,211). happensAt(inactive(wk2gt_1),211). orientation(wk2gt_3,46,211). coords(wk2gt_3,282,267,211).  visible(wk2gt_3,211). happensAt(walking(wk2gt_3),211).
orientation(wk2gt_1,129,212). coords(wk2gt_1,115,53,212).  visible(wk2gt_1,212). happensAt(inactive(wk2gt_1),212). orientation(wk2gt_3,48,212). coords(wk2gt_3,284,270,212).  visible(wk2gt_3,212). happensAt(walking(wk2gt_3),212).
orientation(wk2gt_1,129,213). coords(wk2gt_1,115,53,213).  visible(wk2gt_1,213). happensAt(inactive(wk2gt_1),213). orientation(wk2gt_3,48,213). coords(wk2gt_3,287,272,213).  visible(wk2gt_3,213). happensAt(walking(wk2gt_3),213).
orientation(wk2gt_1,129,214). coords(wk2gt_1,115,53,214).  visible(wk2gt_1,214). happensAt(inactive(wk2gt_1),214). orientation(wk2gt_3,48,214). coords(wk2gt_3,288,273,214).  visible(wk2gt_3,214). happensAt(walking(wk2gt_3),214).
orientation(wk2gt_1,129,215). coords(wk2gt_1,115,53,215).  visible(wk2gt_1,215). happensAt(inactive(wk2gt_1),215). orientation(wk2gt_3,48,215). coords(wk2gt_3,290,273,215).  visible(wk2gt_3,215). happensAt(walking(wk2gt_3),215).
orientation(wk2gt_1,129,216). coords(wk2gt_1,115,53,216).  visible(wk2gt_1,216). happensAt(inactive(wk2gt_1),216). orientation(wk2gt_3,48,216). coords(wk2gt_3,291,273,216).  visible(wk2gt_3,216). happensAt(walking(wk2gt_3),216).
orientation(wk2gt_1,129,217). coords(wk2gt_1,115,53,217).  visible(wk2gt_1,217). happensAt(inactive(wk2gt_1),217). orientation(wk2gt_3,48,217). coords(wk2gt_3,290,273,217).  visible(wk2gt_3,217). happensAt(walking(wk2gt_3),217).
orientation(wk2gt_1,129,218). coords(wk2gt_1,115,53,218).  visible(wk2gt_1,218). happensAt(inactive(wk2gt_1),218). orientation(wk2gt_3,52,218). coords(wk2gt_3,291,273,218).  visible(wk2gt_3,218). happensAt(walking(wk2gt_3),218).
orientation(wk2gt_1,129,219). coords(wk2gt_1,115,53,219).  visible(wk2gt_1,219). happensAt(inactive(wk2gt_1),219). orientation(wk2gt_3,60,219). coords(wk2gt_3,292,273,219).  visible(wk2gt_3,219). happensAt(walking(wk2gt_3),219).
orientation(wk2gt_1,129,220). coords(wk2gt_1,115,53,220).  visible(wk2gt_1,220). happensAt(inactive(wk2gt_1),220). orientation(wk2gt_3,60,220). coords(wk2gt_3,292,273,220).  visible(wk2gt_3,220). happensAt(walking(wk2gt_3),220).
orientation(wk2gt_1,129,221). coords(wk2gt_1,115,53,221).  visible(wk2gt_1,221). happensAt(inactive(wk2gt_1),221). orientation(wk2gt_3,72,221). coords(wk2gt_3,299,274,221).  visible(wk2gt_3,221). happensAt(walking(wk2gt_3),221).
orientation(wk2gt_1,126,222). coords(wk2gt_1,115,53,222).  visible(wk2gt_1,222). happensAt(inactive(wk2gt_1),222). orientation(wk2gt_3,72,222). coords(wk2gt_3,299,276,222).  visible(wk2gt_3,222). happensAt(walking(wk2gt_3),222).
orientation(wk2gt_1,126,223). coords(wk2gt_1,115,53,223).  visible(wk2gt_1,223). happensAt(inactive(wk2gt_1),223). orientation(wk2gt_3,72,223). coords(wk2gt_3,301,277,223).  visible(wk2gt_3,223). happensAt(walking(wk2gt_3),223).
orientation(wk2gt_1,126,224). coords(wk2gt_1,115,53,224).  visible(wk2gt_1,224). happensAt(inactive(wk2gt_1),224). orientation(wk2gt_3,72,224). coords(wk2gt_3,307,279,224).  visible(wk2gt_3,224). happensAt(walking(wk2gt_3),224).
orientation(wk2gt_1,126,225). coords(wk2gt_1,115,53,225).  visible(wk2gt_1,225). happensAt(inactive(wk2gt_1),225). orientation(wk2gt_3,72,225). coords(wk2gt_3,308,281,225).  visible(wk2gt_3,225). happensAt(walking(wk2gt_3),225).
orientation(wk2gt_1,126,226). coords(wk2gt_1,115,53,226).  visible(wk2gt_1,226). happensAt(inactive(wk2gt_1),226). orientation(wk2gt_3,72,226). coords(wk2gt_3,309,283,226).  visible(wk2gt_3,226). happensAt(walking(wk2gt_3),226).
orientation(wk2gt_1,126,227). coords(wk2gt_1,115,53,227).  visible(wk2gt_1,227). happensAt(inactive(wk2gt_1),227). orientation(wk2gt_3,72,227). coords(wk2gt_3,310,285,227).  visible(wk2gt_3,227). happensAt(walking(wk2gt_3),227).
orientation(wk2gt_1,126,228). coords(wk2gt_1,115,53,228).  visible(wk2gt_1,228). happensAt(inactive(wk2gt_1),228). orientation(wk2gt_3,72,228). coords(wk2gt_3,306,285,228).  visible(wk2gt_3,228). happensAt(walking(wk2gt_3),228).
orientation(wk2gt_1,126,229). coords(wk2gt_1,115,53,229).  visible(wk2gt_1,229). happensAt(inactive(wk2gt_1),229). orientation(wk2gt_3,72,229). coords(wk2gt_3,306,285,229).  visible(wk2gt_3,229). happensAt(walking(wk2gt_3),229).
orientation(wk2gt_1,126,230). coords(wk2gt_1,115,53,230).  visible(wk2gt_1,230). happensAt(inactive(wk2gt_1),230). orientation(wk2gt_3,72,230). coords(wk2gt_3,307,285,230).  visible(wk2gt_3,230). happensAt(walking(wk2gt_3),230).
orientation(wk2gt_1,126,231). coords(wk2gt_1,115,53,231).  visible(wk2gt_1,231). happensAt(inactive(wk2gt_1),231). orientation(wk2gt_3,72,231). coords(wk2gt_3,307,286,231).  visible(wk2gt_3,231). happensAt(walking(wk2gt_3),231).
orientation(wk2gt_1,126,232). coords(wk2gt_1,115,53,232).  visible(wk2gt_1,232). happensAt(inactive(wk2gt_1),232). orientation(wk2gt_3,72,232). coords(wk2gt_3,307,286,232).  happensAt(disappear(wk2gt_3),232). happensAt(walking(wk2gt_3),232).
#show holdsAt/2.