
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

	orientation(br1gt_0,47,204). coords(br1gt_0,305,280,204).  visible(br1gt_0,204). happensAt(walking(br1gt_0),204). orientation(br1gt_2,171,204). coords(br1gt_2,86,207,204).  visible(br1gt_2,204). happensAt(inactive(br1gt_2),204).
orientation(br1gt_0,47,205). coords(br1gt_0,305,281,205).  visible(br1gt_0,205). happensAt(walking(br1gt_0),205). orientation(br1gt_2,171,205). coords(br1gt_2,86,207,205).  visible(br1gt_2,205). happensAt(inactive(br1gt_2),205).
orientation(br1gt_0,47,206). coords(br1gt_0,305,281,206).  visible(br1gt_0,206). happensAt(walking(br1gt_0),206). orientation(br1gt_2,171,206). coords(br1gt_2,86,207,206).  visible(br1gt_2,206). happensAt(inactive(br1gt_2),206).
orientation(br1gt_0,47,207). coords(br1gt_0,305,281,207).  visible(br1gt_0,207). happensAt(walking(br1gt_0),207). orientation(br1gt_2,171,207). coords(br1gt_2,86,207,207).  visible(br1gt_2,207). happensAt(inactive(br1gt_2),207).
orientation(br1gt_0,47,208). coords(br1gt_0,305,281,208).  visible(br1gt_0,208). happensAt(walking(br1gt_0),208). orientation(br1gt_2,171,208). coords(br1gt_2,86,207,208).  visible(br1gt_2,208). happensAt(inactive(br1gt_2),208).
orientation(br1gt_0,47,209). coords(br1gt_0,305,281,209).  visible(br1gt_0,209). happensAt(walking(br1gt_0),209). orientation(br1gt_2,171,209). coords(br1gt_2,86,207,209).  visible(br1gt_2,209). happensAt(inactive(br1gt_2),209).
orientation(br1gt_0,47,210). coords(br1gt_0,305,281,210).  visible(br1gt_0,210). happensAt(walking(br1gt_0),210). orientation(br1gt_2,171,210). coords(br1gt_2,86,207,210).  visible(br1gt_2,210). happensAt(inactive(br1gt_2),210).
orientation(br1gt_0,47,211). coords(br1gt_0,305,281,211).  visible(br1gt_0,211). happensAt(walking(br1gt_0),211). orientation(br1gt_2,171,211). coords(br1gt_2,86,207,211).  visible(br1gt_2,211). happensAt(inactive(br1gt_2),211).
orientation(br1gt_0,47,212). coords(br1gt_0,305,282,212).  visible(br1gt_0,212). happensAt(walking(br1gt_0),212). orientation(br1gt_2,171,212). coords(br1gt_2,86,207,212).  visible(br1gt_2,212). happensAt(inactive(br1gt_2),212).
orientation(br1gt_0,47,213). coords(br1gt_0,305,282,213).  visible(br1gt_0,213). happensAt(walking(br1gt_0),213). orientation(br1gt_2,171,213). coords(br1gt_2,86,207,213).  visible(br1gt_2,213). happensAt(inactive(br1gt_2),213).
orientation(br1gt_0,47,214). coords(br1gt_0,305,282,214).  visible(br1gt_0,214). happensAt(walking(br1gt_0),214). orientation(br1gt_2,171,214). coords(br1gt_2,86,207,214).  visible(br1gt_2,214). happensAt(inactive(br1gt_2),214).
orientation(br1gt_0,53,215). coords(br1gt_0,307,282,215).  visible(br1gt_0,215). happensAt(walking(br1gt_0),215). orientation(br1gt_2,171,215). coords(br1gt_2,86,207,215).  visible(br1gt_2,215). happensAt(inactive(br1gt_2),215).
orientation(br1gt_0,53,216). coords(br1gt_0,307,282,216).  visible(br1gt_0,216). happensAt(walking(br1gt_0),216). orientation(br1gt_2,171,216). coords(br1gt_2,86,207,216).  visible(br1gt_2,216). happensAt(inactive(br1gt_2),216).
orientation(br1gt_0,53,217). coords(br1gt_0,307,282,217).  visible(br1gt_0,217). happensAt(walking(br1gt_0),217). orientation(br1gt_2,171,217). coords(br1gt_2,86,207,217).  visible(br1gt_2,217). happensAt(inactive(br1gt_2),217).
orientation(br1gt_0,53,218). coords(br1gt_0,307,282,218).  visible(br1gt_0,218). happensAt(walking(br1gt_0),218). orientation(br1gt_2,171,218). coords(br1gt_2,86,207,218).  visible(br1gt_2,218). happensAt(inactive(br1gt_2),218).
orientation(br1gt_0,53,219). coords(br1gt_0,307,282,219).  visible(br1gt_0,219). happensAt(walking(br1gt_0),219). orientation(br1gt_2,171,219). coords(br1gt_2,86,207,219).  visible(br1gt_2,219). happensAt(inactive(br1gt_2),219).
orientation(br1gt_0,53,220). coords(br1gt_0,307,282,220).  visible(br1gt_0,220). happensAt(walking(br1gt_0),220). orientation(br1gt_2,171,220). coords(br1gt_2,86,207,220).  visible(br1gt_2,220). happensAt(inactive(br1gt_2),220).
orientation(br1gt_0,53,221). coords(br1gt_0,307,282,221).  visible(br1gt_0,221). happensAt(walking(br1gt_0),221). orientation(br1gt_2,171,221). coords(br1gt_2,86,207,221).  visible(br1gt_2,221). happensAt(inactive(br1gt_2),221).
orientation(br1gt_0,53,222). coords(br1gt_0,307,282,222).  visible(br1gt_0,222). happensAt(walking(br1gt_0),222). orientation(br1gt_2,171,222). coords(br1gt_2,86,207,222).  visible(br1gt_2,222). happensAt(inactive(br1gt_2),222).
orientation(br1gt_0,53,223). coords(br1gt_0,307,282,223).  visible(br1gt_0,223). happensAt(walking(br1gt_0),223). orientation(br1gt_2,171,223). coords(br1gt_2,86,207,223).  visible(br1gt_2,223). happensAt(inactive(br1gt_2),223).
orientation(br1gt_0,53,224). coords(br1gt_0,307,282,224).  visible(br1gt_0,224). happensAt(walking(br1gt_0),224). orientation(br1gt_2,171,224). coords(br1gt_2,86,207,224).  visible(br1gt_2,224). happensAt(inactive(br1gt_2),224).
orientation(br1gt_0,53,225). coords(br1gt_0,309,282,225).  visible(br1gt_0,225). happensAt(walking(br1gt_0),225). orientation(br1gt_2,171,225). coords(br1gt_2,86,207,225).  visible(br1gt_2,225). happensAt(inactive(br1gt_2),225).
orientation(br1gt_0,53,226). coords(br1gt_0,309,282,226).  visible(br1gt_0,226). happensAt(walking(br1gt_0),226). orientation(br1gt_2,171,226). coords(br1gt_2,86,207,226).  visible(br1gt_2,226). happensAt(inactive(br1gt_2),226).
orientation(br1gt_0,53,227). coords(br1gt_0,309,282,227).  visible(br1gt_0,227). happensAt(walking(br1gt_0),227). orientation(br1gt_2,171,227). coords(br1gt_2,86,207,227).  visible(br1gt_2,227). happensAt(inactive(br1gt_2),227).
orientation(br1gt_0,53,228). coords(br1gt_0,309,282,228).  visible(br1gt_0,228). happensAt(walking(br1gt_0),228). orientation(br1gt_2,171,228). coords(br1gt_2,86,207,228).  visible(br1gt_2,228). happensAt(inactive(br1gt_2),228).
#show holdsAt/2.