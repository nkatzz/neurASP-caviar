
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

	orientation(br1gt_0,47,204). coords(br1gt_0,305,280,204).  visible(br1gt_0,204). happensAt(walking(br1gt_0),204). orientation(br1gt_1,164,204). coords(br1gt_1,32,157,204).  visible(br1gt_1,204). happensAt(walking(br1gt_1),204).
orientation(br1gt_0,47,205). coords(br1gt_0,305,281,205).  visible(br1gt_0,205). happensAt(walking(br1gt_0),205). orientation(br1gt_1,164,205). coords(br1gt_1,31,157,205).  visible(br1gt_1,205). happensAt(walking(br1gt_1),205).
orientation(br1gt_0,47,206). coords(br1gt_0,305,281,206).  visible(br1gt_0,206). happensAt(walking(br1gt_0),206). orientation(br1gt_1,164,206). coords(br1gt_1,31,157,206).  visible(br1gt_1,206). happensAt(walking(br1gt_1),206).
orientation(br1gt_0,47,207). coords(br1gt_0,305,281,207).  visible(br1gt_0,207). happensAt(walking(br1gt_0),207). orientation(br1gt_1,164,207). coords(br1gt_1,31,157,207).  visible(br1gt_1,207). happensAt(walking(br1gt_1),207).
orientation(br1gt_0,47,208). coords(br1gt_0,305,281,208).  visible(br1gt_0,208). happensAt(walking(br1gt_0),208). orientation(br1gt_1,164,208). coords(br1gt_1,31,157,208).  visible(br1gt_1,208). happensAt(walking(br1gt_1),208).
orientation(br1gt_0,47,209). coords(br1gt_0,305,281,209).  visible(br1gt_0,209). happensAt(walking(br1gt_0),209). orientation(br1gt_1,162,209). coords(br1gt_1,31,158,209).  visible(br1gt_1,209). happensAt(walking(br1gt_1),209).
orientation(br1gt_0,47,210). coords(br1gt_0,305,281,210).  visible(br1gt_0,210). happensAt(walking(br1gt_0),210). orientation(br1gt_1,162,210). coords(br1gt_1,31,158,210).  visible(br1gt_1,210). happensAt(walking(br1gt_1),210).
orientation(br1gt_0,47,211). coords(br1gt_0,305,281,211).  visible(br1gt_0,211). happensAt(walking(br1gt_0),211). orientation(br1gt_1,170,211). coords(br1gt_1,31,158,211).  visible(br1gt_1,211). happensAt(walking(br1gt_1),211).
orientation(br1gt_0,47,212). coords(br1gt_0,305,282,212).  visible(br1gt_0,212). happensAt(walking(br1gt_0),212). orientation(br1gt_1,170,212). coords(br1gt_1,31,158,212).  visible(br1gt_1,212). happensAt(walking(br1gt_1),212).
orientation(br1gt_0,47,213). coords(br1gt_0,305,282,213).  visible(br1gt_0,213). happensAt(walking(br1gt_0),213). orientation(br1gt_1,170,213). coords(br1gt_1,31,158,213).  visible(br1gt_1,213). happensAt(walking(br1gt_1),213).
orientation(br1gt_0,47,214). coords(br1gt_0,305,282,214).  visible(br1gt_0,214). happensAt(walking(br1gt_0),214). orientation(br1gt_1,170,214). coords(br1gt_1,30,159,214).  visible(br1gt_1,214). happensAt(walking(br1gt_1),214).
orientation(br1gt_0,53,215). coords(br1gt_0,307,282,215).  visible(br1gt_0,215). happensAt(walking(br1gt_0),215). orientation(br1gt_1,170,215). coords(br1gt_1,30,159,215).  visible(br1gt_1,215). happensAt(walking(br1gt_1),215).
orientation(br1gt_0,53,216). coords(br1gt_0,307,282,216).  visible(br1gt_0,216). happensAt(walking(br1gt_0),216). orientation(br1gt_1,170,216). coords(br1gt_1,30,159,216).  visible(br1gt_1,216). happensAt(walking(br1gt_1),216).
orientation(br1gt_0,53,217). coords(br1gt_0,307,282,217).  visible(br1gt_0,217). happensAt(walking(br1gt_0),217). orientation(br1gt_1,170,217). coords(br1gt_1,30,159,217).  visible(br1gt_1,217). happensAt(walking(br1gt_1),217).
orientation(br1gt_0,53,218). coords(br1gt_0,307,282,218).  visible(br1gt_0,218). happensAt(walking(br1gt_0),218). orientation(br1gt_1,170,218). coords(br1gt_1,30,159,218).  visible(br1gt_1,218). happensAt(walking(br1gt_1),218).
orientation(br1gt_0,53,219). coords(br1gt_0,307,282,219).  visible(br1gt_0,219). happensAt(walking(br1gt_0),219). orientation(br1gt_1,170,219). coords(br1gt_1,30,159,219).  visible(br1gt_1,219). happensAt(walking(br1gt_1),219).
orientation(br1gt_0,53,220). coords(br1gt_0,307,282,220).  visible(br1gt_0,220). happensAt(walking(br1gt_0),220). orientation(br1gt_1,170,220). coords(br1gt_1,29,160,220).  visible(br1gt_1,220). happensAt(walking(br1gt_1),220).
orientation(br1gt_0,53,221). coords(br1gt_0,307,282,221).  visible(br1gt_0,221). happensAt(walking(br1gt_0),221). orientation(br1gt_1,170,221). coords(br1gt_1,29,161,221).  visible(br1gt_1,221). happensAt(walking(br1gt_1),221).
orientation(br1gt_0,53,222). coords(br1gt_0,307,282,222).  visible(br1gt_0,222). happensAt(walking(br1gt_0),222). orientation(br1gt_1,170,222). coords(br1gt_1,28,160,222).  visible(br1gt_1,222). happensAt(walking(br1gt_1),222).
orientation(br1gt_0,53,223). coords(br1gt_0,307,282,223).  visible(br1gt_0,223). happensAt(walking(br1gt_0),223). orientation(br1gt_1,170,223). coords(br1gt_1,28,161,223).  visible(br1gt_1,223). happensAt(walking(br1gt_1),223).
orientation(br1gt_0,53,224). coords(br1gt_0,307,282,224).  visible(br1gt_0,224). happensAt(walking(br1gt_0),224). orientation(br1gt_1,170,224). coords(br1gt_1,26,161,224).  visible(br1gt_1,224). happensAt(walking(br1gt_1),224).
orientation(br1gt_0,53,225). coords(br1gt_0,309,282,225).  visible(br1gt_0,225). happensAt(walking(br1gt_0),225). orientation(br1gt_1,170,225). coords(br1gt_1,27,163,225).  visible(br1gt_1,225). happensAt(walking(br1gt_1),225).
orientation(br1gt_0,53,226). coords(br1gt_0,309,282,226).  visible(br1gt_0,226). happensAt(walking(br1gt_0),226). orientation(br1gt_1,170,226). coords(br1gt_1,27,162,226).  visible(br1gt_1,226). happensAt(walking(br1gt_1),226).
orientation(br1gt_0,53,227). coords(br1gt_0,309,282,227).  visible(br1gt_0,227). happensAt(walking(br1gt_0),227). orientation(br1gt_1,170,227). coords(br1gt_1,26,163,227).  visible(br1gt_1,227). happensAt(walking(br1gt_1),227).
orientation(br1gt_0,53,228). coords(br1gt_0,309,282,228).  visible(br1gt_0,228). happensAt(walking(br1gt_0),228). orientation(br1gt_1,170,228). coords(br1gt_1,27,162,228).  visible(br1gt_1,228). happensAt(walking(br1gt_1),228).
#show holdsAt/2.