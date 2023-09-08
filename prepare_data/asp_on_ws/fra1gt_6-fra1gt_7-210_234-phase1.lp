
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

	orientation(fra1gt_6,101,210). coords(fra1gt_6,276,13,210).  visible(fra1gt_6,210). happensAt(walking(fra1gt_6),210). orientation(fra1gt_7,127,210). coords(fra1gt_7,163,28,210).  visible(fra1gt_7,210). happensAt(walking(fra1gt_7),210).
orientation(fra1gt_6,101,211). coords(fra1gt_6,275,14,211).  visible(fra1gt_6,211). happensAt(walking(fra1gt_6),211). orientation(fra1gt_7,127,211). coords(fra1gt_7,163,28,211).  visible(fra1gt_7,211). happensAt(walking(fra1gt_7),211).
orientation(fra1gt_6,101,212). coords(fra1gt_6,275,14,212).  visible(fra1gt_6,212). happensAt(walking(fra1gt_6),212). orientation(fra1gt_7,127,212). coords(fra1gt_7,163,29,212).  visible(fra1gt_7,212). happensAt(walking(fra1gt_7),212).
orientation(fra1gt_6,109,213). coords(fra1gt_6,275,14,213).  visible(fra1gt_6,213). happensAt(walking(fra1gt_6),213). orientation(fra1gt_7,127,213). coords(fra1gt_7,163,29,213).  visible(fra1gt_7,213). happensAt(walking(fra1gt_7),213).
orientation(fra1gt_6,109,214). coords(fra1gt_6,272,14,214).  visible(fra1gt_6,214). happensAt(walking(fra1gt_6),214). orientation(fra1gt_7,127,214). coords(fra1gt_7,162,30,214).  visible(fra1gt_7,214). happensAt(walking(fra1gt_7),214).
orientation(fra1gt_6,109,215). coords(fra1gt_6,272,14,215).  visible(fra1gt_6,215). happensAt(walking(fra1gt_6),215). orientation(fra1gt_7,127,215). coords(fra1gt_7,162,30,215).  visible(fra1gt_7,215). happensAt(walking(fra1gt_7),215).
orientation(fra1gt_6,109,216). coords(fra1gt_6,269,14,216).  visible(fra1gt_6,216). happensAt(walking(fra1gt_6),216). orientation(fra1gt_7,127,216). coords(fra1gt_7,162,31,216).  visible(fra1gt_7,216). happensAt(walking(fra1gt_7),216).
orientation(fra1gt_6,109,217). coords(fra1gt_6,268,14,217).  visible(fra1gt_6,217). happensAt(walking(fra1gt_6),217). orientation(fra1gt_7,127,217). coords(fra1gt_7,162,31,217).  visible(fra1gt_7,217). happensAt(walking(fra1gt_7),217).
orientation(fra1gt_6,109,218). coords(fra1gt_6,267,14,218).  visible(fra1gt_6,218). happensAt(walking(fra1gt_6),218). orientation(fra1gt_7,127,218). coords(fra1gt_7,161,31,218).  visible(fra1gt_7,218). happensAt(walking(fra1gt_7),218).
orientation(fra1gt_6,109,219). coords(fra1gt_6,264,15,219).  visible(fra1gt_6,219). happensAt(walking(fra1gt_6),219). orientation(fra1gt_7,127,219). coords(fra1gt_7,160,31,219).  visible(fra1gt_7,219). happensAt(walking(fra1gt_7),219).
orientation(fra1gt_6,109,220). coords(fra1gt_6,262,16,220).  visible(fra1gt_6,220). happensAt(walking(fra1gt_6),220). orientation(fra1gt_7,127,220). coords(fra1gt_7,159,31,220).  visible(fra1gt_7,220). happensAt(walking(fra1gt_7),220).
orientation(fra1gt_6,109,221). coords(fra1gt_6,262,18,221).  visible(fra1gt_6,221). happensAt(walking(fra1gt_6),221). orientation(fra1gt_7,127,221). coords(fra1gt_7,159,31,221).  visible(fra1gt_7,221). happensAt(walking(fra1gt_7),221).
orientation(fra1gt_6,109,222). coords(fra1gt_6,260,19,222).  visible(fra1gt_6,222). happensAt(walking(fra1gt_6),222). orientation(fra1gt_7,127,222). coords(fra1gt_7,159,32,222).  visible(fra1gt_7,222). happensAt(walking(fra1gt_7),222).
orientation(fra1gt_6,109,223). coords(fra1gt_6,259,20,223).  visible(fra1gt_6,223). happensAt(walking(fra1gt_6),223). orientation(fra1gt_7,127,223). coords(fra1gt_7,159,32,223).  visible(fra1gt_7,223). happensAt(walking(fra1gt_7),223).
orientation(fra1gt_6,109,224). coords(fra1gt_6,258,21,224).  visible(fra1gt_6,224). happensAt(walking(fra1gt_6),224). orientation(fra1gt_7,127,224). coords(fra1gt_7,159,32,224).  visible(fra1gt_7,224). happensAt(walking(fra1gt_7),224).
orientation(fra1gt_6,109,225). coords(fra1gt_6,258,21,225).  visible(fra1gt_6,225). happensAt(walking(fra1gt_6),225). orientation(fra1gt_7,127,225). coords(fra1gt_7,159,33,225).  visible(fra1gt_7,225). happensAt(walking(fra1gt_7),225).
orientation(fra1gt_6,109,226). coords(fra1gt_6,258,22,226).  visible(fra1gt_6,226). happensAt(walking(fra1gt_6),226). orientation(fra1gt_7,127,226). coords(fra1gt_7,159,33,226).  visible(fra1gt_7,226). happensAt(walking(fra1gt_7),226).
orientation(fra1gt_6,109,227). coords(fra1gt_6,257,22,227).  visible(fra1gt_6,227). happensAt(walking(fra1gt_6),227). orientation(fra1gt_7,127,227). coords(fra1gt_7,158,33,227).  visible(fra1gt_7,227). happensAt(walking(fra1gt_7),227).
orientation(fra1gt_6,109,228). coords(fra1gt_6,256,22,228).  visible(fra1gt_6,228). happensAt(walking(fra1gt_6),228). orientation(fra1gt_7,127,228). coords(fra1gt_7,158,33,228).  visible(fra1gt_7,228). happensAt(walking(fra1gt_7),228).
orientation(fra1gt_6,109,229). coords(fra1gt_6,255,22,229).  visible(fra1gt_6,229). happensAt(walking(fra1gt_6),229). orientation(fra1gt_7,127,229). coords(fra1gt_7,157,34,229).  visible(fra1gt_7,229). happensAt(walking(fra1gt_7),229).
orientation(fra1gt_6,109,230). coords(fra1gt_6,253,22,230).  visible(fra1gt_6,230). happensAt(walking(fra1gt_6),230). orientation(fra1gt_7,127,230). coords(fra1gt_7,157,34,230).  visible(fra1gt_7,230). happensAt(walking(fra1gt_7),230).
orientation(fra1gt_6,109,231). coords(fra1gt_6,253,22,231).  visible(fra1gt_6,231). happensAt(walking(fra1gt_6),231). orientation(fra1gt_7,127,231). coords(fra1gt_7,157,35,231).  visible(fra1gt_7,231). happensAt(walking(fra1gt_7),231).
orientation(fra1gt_6,109,232). coords(fra1gt_6,252,22,232).  visible(fra1gt_6,232). happensAt(walking(fra1gt_6),232). orientation(fra1gt_7,127,232). coords(fra1gt_7,156,35,232).  visible(fra1gt_7,232). happensAt(walking(fra1gt_7),232).
orientation(fra1gt_6,109,233). coords(fra1gt_6,251,22,233).  visible(fra1gt_6,233). happensAt(walking(fra1gt_6),233). orientation(fra1gt_7,127,233). coords(fra1gt_7,156,35,233).  visible(fra1gt_7,233). happensAt(walking(fra1gt_7),233).
orientation(fra1gt_6,109,234). coords(fra1gt_6,251,22,234).  visible(fra1gt_6,234). happensAt(walking(fra1gt_6),234). orientation(fra1gt_7,127,234). coords(fra1gt_7,156,36,234).  visible(fra1gt_7,234). happensAt(walking(fra1gt_7),234).
#show holdsAt/2.