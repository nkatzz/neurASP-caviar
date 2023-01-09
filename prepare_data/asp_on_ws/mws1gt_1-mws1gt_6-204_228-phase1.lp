
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

	orientation(mws1gt_1,155,204). coords(mws1gt_1,138,133,204).  visible(mws1gt_1,204). happensAt(walking(mws1gt_1),204). orientation(mws1gt_6,140,204). coords(mws1gt_6,60,88,204).  visible(mws1gt_6,204). happensAt(walking(mws1gt_6),204).
orientation(mws1gt_1,155,205). coords(mws1gt_1,138,133,205).  visible(mws1gt_1,205). happensAt(walking(mws1gt_1),205). orientation(mws1gt_6,146,205). coords(mws1gt_6,60,90,205).  visible(mws1gt_6,205). happensAt(walking(mws1gt_6),205).
orientation(mws1gt_1,155,206). coords(mws1gt_1,137,132,206).  visible(mws1gt_1,206). happensAt(walking(mws1gt_1),206). orientation(mws1gt_6,146,206). coords(mws1gt_6,59,91,206).  visible(mws1gt_6,206). happensAt(walking(mws1gt_6),206).
orientation(mws1gt_1,155,207). coords(mws1gt_1,137,132,207).  visible(mws1gt_1,207). happensAt(walking(mws1gt_1),207). orientation(mws1gt_6,146,207). coords(mws1gt_6,59,92,207).  visible(mws1gt_6,207). happensAt(walking(mws1gt_6),207).
orientation(mws1gt_1,155,208). coords(mws1gt_1,137,132,208).  visible(mws1gt_1,208). happensAt(walking(mws1gt_1),208). orientation(mws1gt_6,146,208). coords(mws1gt_6,59,92,208).  visible(mws1gt_6,208). happensAt(walking(mws1gt_6),208).
orientation(mws1gt_1,155,209). coords(mws1gt_1,136,132,209).  visible(mws1gt_1,209). happensAt(walking(mws1gt_1),209). orientation(mws1gt_6,146,209). coords(mws1gt_6,59,94,209).  visible(mws1gt_6,209). happensAt(walking(mws1gt_6),209).
orientation(mws1gt_1,155,210). coords(mws1gt_1,135,132,210).  visible(mws1gt_1,210). happensAt(walking(mws1gt_1),210). orientation(mws1gt_6,146,210). coords(mws1gt_6,59,94,210).  visible(mws1gt_6,210). happensAt(walking(mws1gt_6),210).
orientation(mws1gt_1,155,211). coords(mws1gt_1,134,132,211).  visible(mws1gt_1,211). happensAt(walking(mws1gt_1),211). orientation(mws1gt_6,146,211). coords(mws1gt_6,59,94,211).  visible(mws1gt_6,211). happensAt(walking(mws1gt_6),211).
orientation(mws1gt_1,155,212). coords(mws1gt_1,134,132,212).  visible(mws1gt_1,212). happensAt(walking(mws1gt_1),212). orientation(mws1gt_6,146,212). coords(mws1gt_6,59,95,212).  visible(mws1gt_6,212). happensAt(walking(mws1gt_6),212).
orientation(mws1gt_1,155,213). coords(mws1gt_1,132,132,213).  visible(mws1gt_1,213). happensAt(walking(mws1gt_1),213). orientation(mws1gt_6,146,213). coords(mws1gt_6,58,95,213).  visible(mws1gt_6,213). happensAt(walking(mws1gt_6),213).
orientation(mws1gt_1,155,214). coords(mws1gt_1,131,131,214).  visible(mws1gt_1,214). happensAt(walking(mws1gt_1),214). orientation(mws1gt_6,146,214). coords(mws1gt_6,57,95,214).  visible(mws1gt_6,214). happensAt(walking(mws1gt_6),214).
orientation(mws1gt_1,155,215). coords(mws1gt_1,130,131,215).  visible(mws1gt_1,215). happensAt(walking(mws1gt_1),215). orientation(mws1gt_6,146,215). coords(mws1gt_6,56,96,215).  visible(mws1gt_6,215). happensAt(walking(mws1gt_6),215).
orientation(mws1gt_1,155,216). coords(mws1gt_1,128,132,216).  visible(mws1gt_1,216). happensAt(walking(mws1gt_1),216). orientation(mws1gt_6,146,216). coords(mws1gt_6,55,96,216).  visible(mws1gt_6,216). happensAt(walking(mws1gt_6),216).
orientation(mws1gt_1,155,217). coords(mws1gt_1,127,132,217).  visible(mws1gt_1,217). happensAt(walking(mws1gt_1),217). orientation(mws1gt_6,146,217). coords(mws1gt_6,55,96,217).  visible(mws1gt_6,217). happensAt(walking(mws1gt_6),217).
orientation(mws1gt_1,155,218). coords(mws1gt_1,126,132,218).  visible(mws1gt_1,218). happensAt(walking(mws1gt_1),218). orientation(mws1gt_6,146,218). coords(mws1gt_6,55,97,218).  visible(mws1gt_6,218). happensAt(walking(mws1gt_6),218).
orientation(mws1gt_1,155,219). coords(mws1gt_1,126,132,219).  visible(mws1gt_1,219). happensAt(walking(mws1gt_1),219). orientation(mws1gt_6,146,219). coords(mws1gt_6,55,97,219).  visible(mws1gt_6,219). happensAt(walking(mws1gt_6),219).
orientation(mws1gt_1,155,220). coords(mws1gt_1,126,132,220).  visible(mws1gt_1,220). happensAt(walking(mws1gt_1),220). orientation(mws1gt_6,146,220). coords(mws1gt_6,55,97,220).  visible(mws1gt_6,220). happensAt(walking(mws1gt_6),220).
orientation(mws1gt_1,155,221). coords(mws1gt_1,126,131,221).  visible(mws1gt_1,221). happensAt(walking(mws1gt_1),221). orientation(mws1gt_6,146,221). coords(mws1gt_6,55,98,221).  visible(mws1gt_6,221). happensAt(walking(mws1gt_6),221).
orientation(mws1gt_1,155,222). coords(mws1gt_1,125,131,222).  visible(mws1gt_1,222). happensAt(walking(mws1gt_1),222). orientation(mws1gt_6,146,222). coords(mws1gt_6,55,98,222).  visible(mws1gt_6,222). happensAt(walking(mws1gt_6),222).
orientation(mws1gt_1,155,223). coords(mws1gt_1,125,130,223).  visible(mws1gt_1,223). happensAt(walking(mws1gt_1),223). orientation(mws1gt_6,146,223). coords(mws1gt_6,55,99,223).  visible(mws1gt_6,223). happensAt(walking(mws1gt_6),223).
orientation(mws1gt_1,155,224). coords(mws1gt_1,125,130,224).  visible(mws1gt_1,224). happensAt(walking(mws1gt_1),224). orientation(mws1gt_6,146,224). coords(mws1gt_6,54,99,224).  visible(mws1gt_6,224). happensAt(walking(mws1gt_6),224).
orientation(mws1gt_1,155,225). coords(mws1gt_1,125,130,225).  visible(mws1gt_1,225). happensAt(walking(mws1gt_1),225). orientation(mws1gt_6,146,225). coords(mws1gt_6,53,100,225).  visible(mws1gt_6,225). happensAt(walking(mws1gt_6),225).
orientation(mws1gt_1,155,226). coords(mws1gt_1,125,130,226).  visible(mws1gt_1,226). happensAt(walking(mws1gt_1),226). orientation(mws1gt_6,146,226). coords(mws1gt_6,53,100,226).  visible(mws1gt_6,226). happensAt(walking(mws1gt_6),226).
orientation(mws1gt_1,155,227). coords(mws1gt_1,124,130,227).  visible(mws1gt_1,227). happensAt(walking(mws1gt_1),227). orientation(mws1gt_6,146,227). coords(mws1gt_6,52,101,227).  visible(mws1gt_6,227). happensAt(walking(mws1gt_6),227).
orientation(mws1gt_1,155,228). coords(mws1gt_1,123,129,228).  visible(mws1gt_1,228). happensAt(walking(mws1gt_1),228). orientation(mws1gt_6,146,228). coords(mws1gt_6,52,101,228).  visible(mws1gt_6,228). happensAt(walking(mws1gt_6),228).
#show holdsAt/2.