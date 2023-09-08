
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

	orientation(mws1gt_3,162,204). coords(mws1gt_3,139,155,204).  visible(mws1gt_3,204). happensAt(walking(mws1gt_3),204). orientation(mws1gt_6,140,204). coords(mws1gt_6,60,88,204).  visible(mws1gt_6,204). happensAt(walking(mws1gt_6),204).
orientation(mws1gt_3,162,205). coords(mws1gt_3,139,155,205).  visible(mws1gt_3,205). happensAt(walking(mws1gt_3),205). orientation(mws1gt_6,146,205). coords(mws1gt_6,60,90,205).  visible(mws1gt_6,205). happensAt(walking(mws1gt_6),205).
orientation(mws1gt_3,162,206). coords(mws1gt_3,138,155,206).  visible(mws1gt_3,206). happensAt(walking(mws1gt_3),206). orientation(mws1gt_6,146,206). coords(mws1gt_6,59,91,206).  visible(mws1gt_6,206). happensAt(walking(mws1gt_6),206).
orientation(mws1gt_3,162,207). coords(mws1gt_3,138,155,207).  visible(mws1gt_3,207). happensAt(walking(mws1gt_3),207). orientation(mws1gt_6,146,207). coords(mws1gt_6,59,92,207).  visible(mws1gt_6,207). happensAt(walking(mws1gt_6),207).
orientation(mws1gt_3,162,208). coords(mws1gt_3,137,155,208).  visible(mws1gt_3,208). happensAt(walking(mws1gt_3),208). orientation(mws1gt_6,146,208). coords(mws1gt_6,59,92,208).  visible(mws1gt_6,208). happensAt(walking(mws1gt_6),208).
orientation(mws1gt_3,162,209). coords(mws1gt_3,135,155,209).  visible(mws1gt_3,209). happensAt(walking(mws1gt_3),209). orientation(mws1gt_6,146,209). coords(mws1gt_6,59,94,209).  visible(mws1gt_6,209). happensAt(walking(mws1gt_6),209).
orientation(mws1gt_3,162,210). coords(mws1gt_3,134,155,210).  visible(mws1gt_3,210). happensAt(walking(mws1gt_3),210). orientation(mws1gt_6,146,210). coords(mws1gt_6,59,94,210).  visible(mws1gt_6,210). happensAt(walking(mws1gt_6),210).
orientation(mws1gt_3,162,211). coords(mws1gt_3,132,155,211).  visible(mws1gt_3,211). happensAt(walking(mws1gt_3),211). orientation(mws1gt_6,146,211). coords(mws1gt_6,59,94,211).  visible(mws1gt_6,211). happensAt(walking(mws1gt_6),211).
orientation(mws1gt_3,162,212). coords(mws1gt_3,130,155,212).  visible(mws1gt_3,212). happensAt(walking(mws1gt_3),212). orientation(mws1gt_6,146,212). coords(mws1gt_6,59,95,212).  visible(mws1gt_6,212). happensAt(walking(mws1gt_6),212).
orientation(mws1gt_3,162,213). coords(mws1gt_3,127,155,213).  visible(mws1gt_3,213). happensAt(walking(mws1gt_3),213). orientation(mws1gt_6,146,213). coords(mws1gt_6,58,95,213).  visible(mws1gt_6,213). happensAt(walking(mws1gt_6),213).
orientation(mws1gt_3,162,214). coords(mws1gt_3,126,154,214).  visible(mws1gt_3,214). happensAt(walking(mws1gt_3),214). orientation(mws1gt_6,146,214). coords(mws1gt_6,57,95,214).  visible(mws1gt_6,214). happensAt(walking(mws1gt_6),214).
orientation(mws1gt_3,162,215). coords(mws1gt_3,125,154,215).  visible(mws1gt_3,215). happensAt(walking(mws1gt_3),215). orientation(mws1gt_6,146,215). coords(mws1gt_6,56,96,215).  visible(mws1gt_6,215). happensAt(walking(mws1gt_6),215).
orientation(mws1gt_3,162,216). coords(mws1gt_3,125,154,216).  visible(mws1gt_3,216). happensAt(walking(mws1gt_3),216). orientation(mws1gt_6,146,216). coords(mws1gt_6,55,96,216).  visible(mws1gt_6,216). happensAt(walking(mws1gt_6),216).
orientation(mws1gt_3,162,217). coords(mws1gt_3,125,154,217).  visible(mws1gt_3,217). happensAt(walking(mws1gt_3),217). orientation(mws1gt_6,146,217). coords(mws1gt_6,55,96,217).  visible(mws1gt_6,217). happensAt(walking(mws1gt_6),217).
orientation(mws1gt_3,162,218). coords(mws1gt_3,124,153,218).  visible(mws1gt_3,218). happensAt(walking(mws1gt_3),218). orientation(mws1gt_6,146,218). coords(mws1gt_6,55,97,218).  visible(mws1gt_6,218). happensAt(walking(mws1gt_6),218).
orientation(mws1gt_3,162,219). coords(mws1gt_3,124,153,219).  visible(mws1gt_3,219). happensAt(walking(mws1gt_3),219). orientation(mws1gt_6,146,219). coords(mws1gt_6,55,97,219).  visible(mws1gt_6,219). happensAt(walking(mws1gt_6),219).
orientation(mws1gt_3,162,220). coords(mws1gt_3,123,153,220).  visible(mws1gt_3,220). happensAt(walking(mws1gt_3),220). orientation(mws1gt_6,146,220). coords(mws1gt_6,55,97,220).  visible(mws1gt_6,220). happensAt(walking(mws1gt_6),220).
orientation(mws1gt_3,162,221). coords(mws1gt_3,123,153,221).  visible(mws1gt_3,221). happensAt(walking(mws1gt_3),221). orientation(mws1gt_6,146,221). coords(mws1gt_6,55,98,221).  visible(mws1gt_6,221). happensAt(walking(mws1gt_6),221).
orientation(mws1gt_3,162,222). coords(mws1gt_3,123,153,222).  visible(mws1gt_3,222). happensAt(walking(mws1gt_3),222). orientation(mws1gt_6,146,222). coords(mws1gt_6,55,98,222).  visible(mws1gt_6,222). happensAt(walking(mws1gt_6),222).
orientation(mws1gt_3,162,223). coords(mws1gt_3,122,152,223).  visible(mws1gt_3,223). happensAt(walking(mws1gt_3),223). orientation(mws1gt_6,146,223). coords(mws1gt_6,55,99,223).  visible(mws1gt_6,223). happensAt(walking(mws1gt_6),223).
orientation(mws1gt_3,162,224). coords(mws1gt_3,121,152,224).  visible(mws1gt_3,224). happensAt(walking(mws1gt_3),224). orientation(mws1gt_6,146,224). coords(mws1gt_6,54,99,224).  visible(mws1gt_6,224). happensAt(walking(mws1gt_6),224).
orientation(mws1gt_3,162,225). coords(mws1gt_3,120,151,225).  visible(mws1gt_3,225). happensAt(walking(mws1gt_3),225). orientation(mws1gt_6,146,225). coords(mws1gt_6,53,100,225).  visible(mws1gt_6,225). happensAt(walking(mws1gt_6),225).
orientation(mws1gt_3,162,226). coords(mws1gt_3,118,150,226).  visible(mws1gt_3,226). happensAt(walking(mws1gt_3),226). orientation(mws1gt_6,146,226). coords(mws1gt_6,53,100,226).  visible(mws1gt_6,226). happensAt(walking(mws1gt_6),226).
orientation(mws1gt_3,162,227). coords(mws1gt_3,116,149,227).  visible(mws1gt_3,227). happensAt(walking(mws1gt_3),227). orientation(mws1gt_6,146,227). coords(mws1gt_6,52,101,227).  visible(mws1gt_6,227). happensAt(walking(mws1gt_6),227).
orientation(mws1gt_3,162,228). coords(mws1gt_3,115,149,228).  visible(mws1gt_3,228). happensAt(walking(mws1gt_3),228). orientation(mws1gt_6,146,228). coords(mws1gt_6,52,101,228).  visible(mws1gt_6,228). happensAt(walking(mws1gt_6),228).
#show holdsAt/2.