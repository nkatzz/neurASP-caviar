
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

	orientation(wk1gt_4,168,207). coords(wk1gt_4,47,181,207).  visible(wk1gt_4,207). happensAt(inactive(wk1gt_4),207). orientation(wk1gt_5,162,207). coords(wk1gt_5,50,166,207).  visible(wk1gt_5,207). happensAt(inactive(wk1gt_5),207).
orientation(wk1gt_4,164,208). coords(wk1gt_4,47,181,208).  visible(wk1gt_4,208). happensAt(inactive(wk1gt_4),208). orientation(wk1gt_5,162,208). coords(wk1gt_5,50,166,208).  visible(wk1gt_5,208). happensAt(inactive(wk1gt_5),208).
orientation(wk1gt_4,164,209). coords(wk1gt_4,47,181,209).  visible(wk1gt_4,209). happensAt(inactive(wk1gt_4),209). orientation(wk1gt_5,162,209). coords(wk1gt_5,50,166,209).  visible(wk1gt_5,209). happensAt(inactive(wk1gt_5),209).
orientation(wk1gt_4,164,210). coords(wk1gt_4,47,181,210).  visible(wk1gt_4,210). happensAt(inactive(wk1gt_4),210). orientation(wk1gt_5,162,210). coords(wk1gt_5,50,166,210).  visible(wk1gt_5,210). happensAt(inactive(wk1gt_5),210).
orientation(wk1gt_4,163,211). coords(wk1gt_4,47,181,211).  visible(wk1gt_4,211). happensAt(inactive(wk1gt_4),211). orientation(wk1gt_5,162,211). coords(wk1gt_5,50,166,211).  visible(wk1gt_5,211). happensAt(inactive(wk1gt_5),211).
orientation(wk1gt_4,164,212). coords(wk1gt_4,47,181,212).  visible(wk1gt_4,212). happensAt(inactive(wk1gt_4),212). orientation(wk1gt_5,162,212). coords(wk1gt_5,50,166,212).  visible(wk1gt_5,212). happensAt(inactive(wk1gt_5),212).
orientation(wk1gt_4,165,213). coords(wk1gt_4,47,181,213).  visible(wk1gt_4,213). happensAt(inactive(wk1gt_4),213). orientation(wk1gt_5,162,213). coords(wk1gt_5,50,166,213).  visible(wk1gt_5,213). happensAt(inactive(wk1gt_5),213).
orientation(wk1gt_4,163,214). coords(wk1gt_4,47,181,214).  visible(wk1gt_4,214). happensAt(inactive(wk1gt_4),214). orientation(wk1gt_5,162,214). coords(wk1gt_5,50,166,214).  visible(wk1gt_5,214). happensAt(inactive(wk1gt_5),214).
orientation(wk1gt_4,163,215). coords(wk1gt_4,47,181,215).  visible(wk1gt_4,215). happensAt(inactive(wk1gt_4),215). orientation(wk1gt_5,162,215). coords(wk1gt_5,50,166,215).  visible(wk1gt_5,215). happensAt(inactive(wk1gt_5),215).
orientation(wk1gt_4,165,216). coords(wk1gt_4,47,181,216).  visible(wk1gt_4,216). happensAt(inactive(wk1gt_4),216). orientation(wk1gt_5,162,216). coords(wk1gt_5,50,166,216).  visible(wk1gt_5,216). happensAt(inactive(wk1gt_5),216).
orientation(wk1gt_4,164,217). coords(wk1gt_4,47,181,217).  visible(wk1gt_4,217). happensAt(inactive(wk1gt_4),217). orientation(wk1gt_5,162,217). coords(wk1gt_5,50,166,217).  visible(wk1gt_5,217). happensAt(inactive(wk1gt_5),217).
orientation(wk1gt_4,166,218). coords(wk1gt_4,47,181,218).  visible(wk1gt_4,218). happensAt(inactive(wk1gt_4),218). orientation(wk1gt_5,162,218). coords(wk1gt_5,50,166,218).  visible(wk1gt_5,218). happensAt(inactive(wk1gt_5),218).
orientation(wk1gt_4,164,219). coords(wk1gt_4,47,181,219).  visible(wk1gt_4,219). happensAt(inactive(wk1gt_4),219). orientation(wk1gt_5,162,219). coords(wk1gt_5,50,166,219).  visible(wk1gt_5,219). happensAt(inactive(wk1gt_5),219).
orientation(wk1gt_4,166,220). coords(wk1gt_4,47,181,220).  visible(wk1gt_4,220). happensAt(inactive(wk1gt_4),220). orientation(wk1gt_5,162,220). coords(wk1gt_5,50,166,220).  visible(wk1gt_5,220). happensAt(inactive(wk1gt_5),220).
orientation(wk1gt_4,164,221). coords(wk1gt_4,47,181,221).  visible(wk1gt_4,221). happensAt(inactive(wk1gt_4),221). orientation(wk1gt_5,162,221). coords(wk1gt_5,50,166,221).  visible(wk1gt_5,221). happensAt(inactive(wk1gt_5),221).
orientation(wk1gt_4,168,222). coords(wk1gt_4,47,181,222).  visible(wk1gt_4,222). happensAt(inactive(wk1gt_4),222). orientation(wk1gt_5,162,222). coords(wk1gt_5,50,166,222).  visible(wk1gt_5,222). happensAt(inactive(wk1gt_5),222).
orientation(wk1gt_4,163,223). coords(wk1gt_4,47,181,223).  visible(wk1gt_4,223). happensAt(inactive(wk1gt_4),223). orientation(wk1gt_5,162,223). coords(wk1gt_5,50,166,223).  visible(wk1gt_5,223). happensAt(inactive(wk1gt_5),223).
orientation(wk1gt_4,165,224). coords(wk1gt_4,47,181,224).  visible(wk1gt_4,224). happensAt(inactive(wk1gt_4),224). orientation(wk1gt_5,162,224). coords(wk1gt_5,50,166,224).  visible(wk1gt_5,224). happensAt(inactive(wk1gt_5),224).
orientation(wk1gt_4,165,225). coords(wk1gt_4,47,181,225).  visible(wk1gt_4,225). happensAt(inactive(wk1gt_4),225). orientation(wk1gt_5,162,225). coords(wk1gt_5,50,166,225).  visible(wk1gt_5,225). happensAt(inactive(wk1gt_5),225).
orientation(wk1gt_4,166,226). coords(wk1gt_4,47,181,226).  visible(wk1gt_4,226). happensAt(inactive(wk1gt_4),226). orientation(wk1gt_5,162,226). coords(wk1gt_5,50,166,226).  visible(wk1gt_5,226). happensAt(inactive(wk1gt_5),226).
orientation(wk1gt_4,164,227). coords(wk1gt_4,47,181,227).  visible(wk1gt_4,227). happensAt(inactive(wk1gt_4),227). orientation(wk1gt_5,162,227). coords(wk1gt_5,50,166,227).  visible(wk1gt_5,227). happensAt(inactive(wk1gt_5),227).
orientation(wk1gt_4,164,228). coords(wk1gt_4,47,181,228).  visible(wk1gt_4,228). happensAt(inactive(wk1gt_4),228). orientation(wk1gt_5,162,228). coords(wk1gt_5,50,166,228).  visible(wk1gt_5,228). happensAt(inactive(wk1gt_5),228).
orientation(wk1gt_4,163,229). coords(wk1gt_4,47,181,229).  visible(wk1gt_4,229). happensAt(inactive(wk1gt_4),229). orientation(wk1gt_5,162,229). coords(wk1gt_5,50,166,229).  visible(wk1gt_5,229). happensAt(inactive(wk1gt_5),229).
orientation(wk1gt_4,159,230). coords(wk1gt_4,47,181,230).  visible(wk1gt_4,230). happensAt(inactive(wk1gt_4),230). orientation(wk1gt_5,162,230). coords(wk1gt_5,50,166,230).  visible(wk1gt_5,230). happensAt(inactive(wk1gt_5),230).
orientation(wk1gt_4,162,231). coords(wk1gt_4,47,181,231).  visible(wk1gt_4,231). happensAt(inactive(wk1gt_4),231). orientation(wk1gt_5,162,231). coords(wk1gt_5,50,166,231).  visible(wk1gt_5,231). happensAt(inactive(wk1gt_5),231).
#show holdsAt/2.