
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

	orientation(wk2gt_1,129,204). coords(wk2gt_1,115,53,204).  visible(wk2gt_1,204). happensAt(inactive(wk2gt_1),204). orientation(wk2gt_2,141,204). coords(wk2gt_2,98,62,204).  visible(wk2gt_2,204). happensAt(inactive(wk2gt_2),204).
orientation(wk2gt_1,129,205). coords(wk2gt_1,115,53,205).  visible(wk2gt_1,205). happensAt(inactive(wk2gt_1),205). orientation(wk2gt_2,141,205). coords(wk2gt_2,98,62,205).  visible(wk2gt_2,205). happensAt(inactive(wk2gt_2),205).
orientation(wk2gt_1,129,206). coords(wk2gt_1,115,53,206).  visible(wk2gt_1,206). happensAt(inactive(wk2gt_1),206). orientation(wk2gt_2,141,206). coords(wk2gt_2,98,62,206).  visible(wk2gt_2,206). happensAt(inactive(wk2gt_2),206).
orientation(wk2gt_1,129,207). coords(wk2gt_1,115,53,207).  visible(wk2gt_1,207). happensAt(inactive(wk2gt_1),207). orientation(wk2gt_2,141,207). coords(wk2gt_2,98,62,207).  visible(wk2gt_2,207). happensAt(inactive(wk2gt_2),207).
orientation(wk2gt_1,129,208). coords(wk2gt_1,115,53,208).  visible(wk2gt_1,208). happensAt(inactive(wk2gt_1),208). orientation(wk2gt_2,141,208). coords(wk2gt_2,98,62,208).  visible(wk2gt_2,208). happensAt(inactive(wk2gt_2),208).
orientation(wk2gt_1,129,209). coords(wk2gt_1,115,53,209).  visible(wk2gt_1,209). happensAt(inactive(wk2gt_1),209). orientation(wk2gt_2,141,209). coords(wk2gt_2,98,62,209).  visible(wk2gt_2,209). happensAt(inactive(wk2gt_2),209).
orientation(wk2gt_1,129,210). coords(wk2gt_1,115,53,210).  visible(wk2gt_1,210). happensAt(inactive(wk2gt_1),210). orientation(wk2gt_2,141,210). coords(wk2gt_2,98,62,210).  visible(wk2gt_2,210). happensAt(inactive(wk2gt_2),210).
orientation(wk2gt_1,129,211). coords(wk2gt_1,115,53,211).  visible(wk2gt_1,211). happensAt(inactive(wk2gt_1),211). orientation(wk2gt_2,141,211). coords(wk2gt_2,98,62,211).  visible(wk2gt_2,211). happensAt(inactive(wk2gt_2),211).
orientation(wk2gt_1,129,212). coords(wk2gt_1,115,53,212).  visible(wk2gt_1,212). happensAt(inactive(wk2gt_1),212). orientation(wk2gt_2,141,212). coords(wk2gt_2,98,62,212).  visible(wk2gt_2,212). happensAt(inactive(wk2gt_2),212).
orientation(wk2gt_1,129,213). coords(wk2gt_1,115,53,213).  visible(wk2gt_1,213). happensAt(inactive(wk2gt_1),213). orientation(wk2gt_2,141,213). coords(wk2gt_2,98,62,213).  visible(wk2gt_2,213). happensAt(inactive(wk2gt_2),213).
orientation(wk2gt_1,129,214). coords(wk2gt_1,115,53,214).  visible(wk2gt_1,214). happensAt(inactive(wk2gt_1),214). orientation(wk2gt_2,141,214). coords(wk2gt_2,98,62,214).  visible(wk2gt_2,214). happensAt(inactive(wk2gt_2),214).
orientation(wk2gt_1,129,215). coords(wk2gt_1,115,53,215).  visible(wk2gt_1,215). happensAt(inactive(wk2gt_1),215). orientation(wk2gt_2,141,215). coords(wk2gt_2,98,62,215).  visible(wk2gt_2,215). happensAt(inactive(wk2gt_2),215).
orientation(wk2gt_1,129,216). coords(wk2gt_1,115,53,216).  visible(wk2gt_1,216). happensAt(inactive(wk2gt_1),216). orientation(wk2gt_2,141,216). coords(wk2gt_2,98,62,216).  visible(wk2gt_2,216). happensAt(inactive(wk2gt_2),216).
orientation(wk2gt_1,129,217). coords(wk2gt_1,115,53,217).  visible(wk2gt_1,217). happensAt(inactive(wk2gt_1),217). orientation(wk2gt_2,141,217). coords(wk2gt_2,98,62,217).  visible(wk2gt_2,217). happensAt(inactive(wk2gt_2),217).
orientation(wk2gt_1,129,218). coords(wk2gt_1,115,53,218).  visible(wk2gt_1,218). happensAt(inactive(wk2gt_1),218). orientation(wk2gt_2,141,218). coords(wk2gt_2,98,62,218).  visible(wk2gt_2,218). happensAt(inactive(wk2gt_2),218).
orientation(wk2gt_1,129,219). coords(wk2gt_1,115,53,219).  visible(wk2gt_1,219). happensAt(inactive(wk2gt_1),219). orientation(wk2gt_2,141,219). coords(wk2gt_2,98,62,219).  visible(wk2gt_2,219). happensAt(inactive(wk2gt_2),219).
orientation(wk2gt_1,129,220). coords(wk2gt_1,115,53,220).  visible(wk2gt_1,220). happensAt(inactive(wk2gt_1),220). orientation(wk2gt_2,141,220). coords(wk2gt_2,98,62,220).  visible(wk2gt_2,220). happensAt(inactive(wk2gt_2),220).
orientation(wk2gt_1,129,221). coords(wk2gt_1,115,53,221).  visible(wk2gt_1,221). happensAt(inactive(wk2gt_1),221). orientation(wk2gt_2,141,221). coords(wk2gt_2,98,62,221).  visible(wk2gt_2,221). happensAt(inactive(wk2gt_2),221).
orientation(wk2gt_1,126,222). coords(wk2gt_1,115,53,222).  visible(wk2gt_1,222). happensAt(inactive(wk2gt_1),222). orientation(wk2gt_2,141,222). coords(wk2gt_2,98,62,222).  visible(wk2gt_2,222). happensAt(inactive(wk2gt_2),222).
orientation(wk2gt_1,126,223). coords(wk2gt_1,115,53,223).  visible(wk2gt_1,223). happensAt(inactive(wk2gt_1),223). orientation(wk2gt_2,141,223). coords(wk2gt_2,98,62,223).  visible(wk2gt_2,223). happensAt(inactive(wk2gt_2),223).
orientation(wk2gt_1,126,224). coords(wk2gt_1,115,53,224).  visible(wk2gt_1,224). happensAt(inactive(wk2gt_1),224). orientation(wk2gt_2,141,224). coords(wk2gt_2,98,62,224).  visible(wk2gt_2,224). happensAt(inactive(wk2gt_2),224).
orientation(wk2gt_1,126,225). coords(wk2gt_1,115,53,225).  visible(wk2gt_1,225). happensAt(inactive(wk2gt_1),225). orientation(wk2gt_2,141,225). coords(wk2gt_2,98,62,225).  visible(wk2gt_2,225). happensAt(inactive(wk2gt_2),225).
orientation(wk2gt_1,126,226). coords(wk2gt_1,115,53,226).  visible(wk2gt_1,226). happensAt(inactive(wk2gt_1),226). orientation(wk2gt_2,141,226). coords(wk2gt_2,98,62,226).  visible(wk2gt_2,226). happensAt(inactive(wk2gt_2),226).
orientation(wk2gt_1,126,227). coords(wk2gt_1,115,53,227).  visible(wk2gt_1,227). happensAt(inactive(wk2gt_1),227). orientation(wk2gt_2,141,227). coords(wk2gt_2,98,62,227).  visible(wk2gt_2,227). happensAt(inactive(wk2gt_2),227).
orientation(wk2gt_1,126,228). coords(wk2gt_1,115,53,228).  visible(wk2gt_1,228). happensAt(inactive(wk2gt_1),228). orientation(wk2gt_2,141,228). coords(wk2gt_2,98,62,228).  visible(wk2gt_2,228). happensAt(inactive(wk2gt_2),228).
#show holdsAt/2.