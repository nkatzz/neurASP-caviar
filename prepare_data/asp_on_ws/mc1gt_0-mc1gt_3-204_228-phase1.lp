
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

	orientation(mc1gt_0,147,204). coords(mc1gt_0,140,118,204).  visible(mc1gt_0,204). happensAt(walking(mc1gt_0),204). orientation(mc1gt_3,146,204). coords(mc1gt_3,133,84,204).  visible(mc1gt_3,204). happensAt(walking(mc1gt_3),204).
orientation(mc1gt_0,147,205). coords(mc1gt_0,141,119,205).  visible(mc1gt_0,205). happensAt(walking(mc1gt_0),205). orientation(mc1gt_3,146,205). coords(mc1gt_3,134,84,205).  visible(mc1gt_3,205). happensAt(walking(mc1gt_3),205).
orientation(mc1gt_0,147,206). coords(mc1gt_0,141,119,206).  visible(mc1gt_0,206). happensAt(walking(mc1gt_0),206). orientation(mc1gt_3,146,206). coords(mc1gt_3,136,84,206).  visible(mc1gt_3,206). happensAt(walking(mc1gt_3),206).
orientation(mc1gt_0,147,207). coords(mc1gt_0,142,119,207).  visible(mc1gt_0,207). happensAt(walking(mc1gt_0),207). orientation(mc1gt_3,146,207). coords(mc1gt_3,137,82,207).  visible(mc1gt_3,207). happensAt(walking(mc1gt_3),207).
orientation(mc1gt_0,147,208). coords(mc1gt_0,142,119,208).  visible(mc1gt_0,208). happensAt(walking(mc1gt_0),208). orientation(mc1gt_3,146,208). coords(mc1gt_3,138,82,208).  visible(mc1gt_3,208). happensAt(walking(mc1gt_3),208).
orientation(mc1gt_0,147,209). coords(mc1gt_0,142,119,209).  visible(mc1gt_0,209). happensAt(walking(mc1gt_0),209). orientation(mc1gt_3,146,209). coords(mc1gt_3,138,83,209).  visible(mc1gt_3,209). happensAt(walking(mc1gt_3),209).
orientation(mc1gt_0,147,210). coords(mc1gt_0,143,119,210).  visible(mc1gt_0,210). happensAt(walking(mc1gt_0),210). orientation(mc1gt_3,146,210). coords(mc1gt_3,139,83,210).  visible(mc1gt_3,210). happensAt(walking(mc1gt_3),210).
orientation(mc1gt_0,147,211). coords(mc1gt_0,143,120,211).  visible(mc1gt_0,211). happensAt(walking(mc1gt_0),211). orientation(mc1gt_3,146,211). coords(mc1gt_3,141,83,211).  visible(mc1gt_3,211). happensAt(walking(mc1gt_3),211).
orientation(mc1gt_0,147,212). coords(mc1gt_0,143,121,212).  visible(mc1gt_0,212). happensAt(walking(mc1gt_0),212). orientation(mc1gt_3,146,212). coords(mc1gt_3,142,83,212).  visible(mc1gt_3,212). happensAt(walking(mc1gt_3),212).
orientation(mc1gt_0,147,213). coords(mc1gt_0,144,122,213).  visible(mc1gt_0,213). happensAt(walking(mc1gt_0),213). orientation(mc1gt_3,146,213). coords(mc1gt_3,142,83,213).  visible(mc1gt_3,213). happensAt(walking(mc1gt_3),213).
orientation(mc1gt_0,147,214). coords(mc1gt_0,144,123,214).  visible(mc1gt_0,214). happensAt(walking(mc1gt_0),214). orientation(mc1gt_3,146,214). coords(mc1gt_3,143,83,214).  visible(mc1gt_3,214). happensAt(walking(mc1gt_3),214).
orientation(mc1gt_0,147,215). coords(mc1gt_0,146,125,215).  visible(mc1gt_0,215). happensAt(walking(mc1gt_0),215). orientation(mc1gt_3,146,215). coords(mc1gt_3,144,84,215).  visible(mc1gt_3,215). happensAt(walking(mc1gt_3),215).
orientation(mc1gt_0,147,216). coords(mc1gt_0,147,126,216).  visible(mc1gt_0,216). happensAt(walking(mc1gt_0),216). orientation(mc1gt_3,146,216). coords(mc1gt_3,145,84,216).  visible(mc1gt_3,216). happensAt(walking(mc1gt_3),216).
orientation(mc1gt_0,147,217). coords(mc1gt_0,149,126,217).  visible(mc1gt_0,217). happensAt(walking(mc1gt_0),217). orientation(mc1gt_3,146,217). coords(mc1gt_3,148,85,217).  visible(mc1gt_3,217). happensAt(walking(mc1gt_3),217).
orientation(mc1gt_0,147,218). coords(mc1gt_0,150,127,218).  visible(mc1gt_0,218). happensAt(walking(mc1gt_0),218). orientation(mc1gt_3,146,218). coords(mc1gt_3,151,86,218).  visible(mc1gt_3,218). happensAt(walking(mc1gt_3),218).
orientation(mc1gt_0,147,219). coords(mc1gt_0,151,128,219).  visible(mc1gt_0,219). happensAt(walking(mc1gt_0),219). orientation(mc1gt_3,146,219). coords(mc1gt_3,153,87,219).  visible(mc1gt_3,219). happensAt(walking(mc1gt_3),219).
orientation(mc1gt_0,147,220). coords(mc1gt_0,151,128,220).  visible(mc1gt_0,220). happensAt(walking(mc1gt_0),220). orientation(mc1gt_3,146,220). coords(mc1gt_3,154,87,220).  visible(mc1gt_3,220). happensAt(walking(mc1gt_3),220).
orientation(mc1gt_0,147,221). coords(mc1gt_0,152,128,221).  visible(mc1gt_0,221). happensAt(walking(mc1gt_0),221). orientation(mc1gt_3,146,221). coords(mc1gt_3,154,87,221).  visible(mc1gt_3,221). happensAt(walking(mc1gt_3),221).
orientation(mc1gt_0,147,222). coords(mc1gt_0,152,129,222).  visible(mc1gt_0,222). happensAt(walking(mc1gt_0),222). orientation(mc1gt_3,146,222). coords(mc1gt_3,154,87,222).  visible(mc1gt_3,222). happensAt(walking(mc1gt_3),222).
orientation(mc1gt_0,147,223). coords(mc1gt_0,153,129,223).  visible(mc1gt_0,223). happensAt(walking(mc1gt_0),223). orientation(mc1gt_3,146,223). coords(mc1gt_3,155,88,223).  visible(mc1gt_3,223). happensAt(walking(mc1gt_3),223).
orientation(mc1gt_0,147,224). coords(mc1gt_0,153,130,224).  visible(mc1gt_0,224). happensAt(walking(mc1gt_0),224). orientation(mc1gt_3,146,224). coords(mc1gt_3,155,88,224).  visible(mc1gt_3,224). happensAt(walking(mc1gt_3),224).
orientation(mc1gt_0,147,225). coords(mc1gt_0,153,130,225).  visible(mc1gt_0,225). happensAt(walking(mc1gt_0),225). orientation(mc1gt_3,146,225). coords(mc1gt_3,157,88,225).  visible(mc1gt_3,225). happensAt(walking(mc1gt_3),225).
orientation(mc1gt_0,147,226). coords(mc1gt_0,154,130,226).  visible(mc1gt_0,226). happensAt(walking(mc1gt_0),226). orientation(mc1gt_3,146,226). coords(mc1gt_3,160,88,226).  visible(mc1gt_3,226). happensAt(walking(mc1gt_3),226).
orientation(mc1gt_0,147,227). coords(mc1gt_0,155,131,227).  visible(mc1gt_0,227). happensAt(walking(mc1gt_0),227). orientation(mc1gt_3,146,227). coords(mc1gt_3,163,88,227).  visible(mc1gt_3,227). happensAt(walking(mc1gt_3),227).
orientation(mc1gt_0,147,228). coords(mc1gt_0,157,131,228).  visible(mc1gt_0,228). happensAt(walking(mc1gt_0),228). orientation(mc1gt_3,146,228). coords(mc1gt_3,165,88,228).  visible(mc1gt_3,228). happensAt(walking(mc1gt_3),228).
#show holdsAt/2.