
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

	orientation(mc1gt_2,148,204). coords(mc1gt_2,106,121,204).  visible(mc1gt_2,204). happensAt(walking(mc1gt_2),204). orientation(mc1gt_3,146,204). coords(mc1gt_3,133,84,204).  visible(mc1gt_3,204). happensAt(walking(mc1gt_3),204).
orientation(mc1gt_2,148,205). coords(mc1gt_2,106,122,205).  visible(mc1gt_2,205). happensAt(walking(mc1gt_2),205). orientation(mc1gt_3,146,205). coords(mc1gt_3,134,84,205).  visible(mc1gt_3,205). happensAt(walking(mc1gt_3),205).
orientation(mc1gt_2,148,206). coords(mc1gt_2,108,122,206).  visible(mc1gt_2,206). happensAt(walking(mc1gt_2),206). orientation(mc1gt_3,146,206). coords(mc1gt_3,136,84,206).  visible(mc1gt_3,206). happensAt(walking(mc1gt_3),206).
orientation(mc1gt_2,148,207). coords(mc1gt_2,109,123,207).  visible(mc1gt_2,207). happensAt(walking(mc1gt_2),207). orientation(mc1gt_3,146,207). coords(mc1gt_3,137,82,207).  visible(mc1gt_3,207). happensAt(walking(mc1gt_3),207).
orientation(mc1gt_2,148,208). coords(mc1gt_2,109,123,208).  visible(mc1gt_2,208). happensAt(walking(mc1gt_2),208). orientation(mc1gt_3,146,208). coords(mc1gt_3,138,82,208).  visible(mc1gt_3,208). happensAt(walking(mc1gt_3),208).
orientation(mc1gt_2,148,209). coords(mc1gt_2,110,124,209).  visible(mc1gt_2,209). happensAt(walking(mc1gt_2),209). orientation(mc1gt_3,146,209). coords(mc1gt_3,138,83,209).  visible(mc1gt_3,209). happensAt(walking(mc1gt_3),209).
orientation(mc1gt_2,148,210). coords(mc1gt_2,110,124,210).  visible(mc1gt_2,210). happensAt(walking(mc1gt_2),210). orientation(mc1gt_3,146,210). coords(mc1gt_3,139,83,210).  visible(mc1gt_3,210). happensAt(walking(mc1gt_3),210).
orientation(mc1gt_2,148,211). coords(mc1gt_2,110,124,211).  visible(mc1gt_2,211). happensAt(walking(mc1gt_2),211). orientation(mc1gt_3,146,211). coords(mc1gt_3,141,83,211).  visible(mc1gt_3,211). happensAt(walking(mc1gt_3),211).
orientation(mc1gt_2,148,212). coords(mc1gt_2,111,124,212).  visible(mc1gt_2,212). happensAt(walking(mc1gt_2),212). orientation(mc1gt_3,146,212). coords(mc1gt_3,142,83,212).  visible(mc1gt_3,212). happensAt(walking(mc1gt_3),212).
orientation(mc1gt_2,148,213). coords(mc1gt_2,111,125,213).  visible(mc1gt_2,213). happensAt(walking(mc1gt_2),213). orientation(mc1gt_3,146,213). coords(mc1gt_3,142,83,213).  visible(mc1gt_3,213). happensAt(walking(mc1gt_3),213).
orientation(mc1gt_2,148,214). coords(mc1gt_2,111,125,214).  visible(mc1gt_2,214). happensAt(walking(mc1gt_2),214). orientation(mc1gt_3,146,214). coords(mc1gt_3,143,83,214).  visible(mc1gt_3,214). happensAt(walking(mc1gt_3),214).
orientation(mc1gt_2,152,215). coords(mc1gt_2,113,125,215).  visible(mc1gt_2,215). happensAt(walking(mc1gt_2),215). orientation(mc1gt_3,146,215). coords(mc1gt_3,144,84,215).  visible(mc1gt_3,215). happensAt(walking(mc1gt_3),215).
orientation(mc1gt_2,152,216). coords(mc1gt_2,115,125,216).  visible(mc1gt_2,216). happensAt(walking(mc1gt_2),216). orientation(mc1gt_3,146,216). coords(mc1gt_3,145,84,216).  visible(mc1gt_3,216). happensAt(walking(mc1gt_3),216).
orientation(mc1gt_2,152,217). coords(mc1gt_2,116,125,217).  visible(mc1gt_2,217). happensAt(walking(mc1gt_2),217). orientation(mc1gt_3,146,217). coords(mc1gt_3,148,85,217).  visible(mc1gt_3,217). happensAt(walking(mc1gt_3),217).
orientation(mc1gt_2,152,218). coords(mc1gt_2,117,125,218).  visible(mc1gt_2,218). happensAt(walking(mc1gt_2),218). orientation(mc1gt_3,146,218). coords(mc1gt_3,151,86,218).  visible(mc1gt_3,218). happensAt(walking(mc1gt_3),218).
orientation(mc1gt_2,152,219). coords(mc1gt_2,119,126,219).  visible(mc1gt_2,219). happensAt(walking(mc1gt_2),219). orientation(mc1gt_3,146,219). coords(mc1gt_3,153,87,219).  visible(mc1gt_3,219). happensAt(walking(mc1gt_3),219).
orientation(mc1gt_2,152,220). coords(mc1gt_2,119,126,220).  visible(mc1gt_2,220). happensAt(walking(mc1gt_2),220). orientation(mc1gt_3,146,220). coords(mc1gt_3,154,87,220).  visible(mc1gt_3,220). happensAt(walking(mc1gt_3),220).
orientation(mc1gt_2,152,221). coords(mc1gt_2,121,127,221).  visible(mc1gt_2,221). happensAt(walking(mc1gt_2),221). orientation(mc1gt_3,146,221). coords(mc1gt_3,154,87,221).  visible(mc1gt_3,221). happensAt(walking(mc1gt_3),221).
orientation(mc1gt_2,152,222). coords(mc1gt_2,121,127,222).  visible(mc1gt_2,222). happensAt(walking(mc1gt_2),222). orientation(mc1gt_3,146,222). coords(mc1gt_3,154,87,222).  visible(mc1gt_3,222). happensAt(walking(mc1gt_3),222).
orientation(mc1gt_2,152,223). coords(mc1gt_2,122,127,223).  visible(mc1gt_2,223). happensAt(walking(mc1gt_2),223). orientation(mc1gt_3,146,223). coords(mc1gt_3,155,88,223).  visible(mc1gt_3,223). happensAt(walking(mc1gt_3),223).
orientation(mc1gt_2,149,224). coords(mc1gt_2,122,127,224).  visible(mc1gt_2,224). happensAt(walking(mc1gt_2),224). orientation(mc1gt_3,146,224). coords(mc1gt_3,155,88,224).  visible(mc1gt_3,224). happensAt(walking(mc1gt_3),224).
orientation(mc1gt_2,148,225). coords(mc1gt_2,123,127,225).  visible(mc1gt_2,225). happensAt(walking(mc1gt_2),225). orientation(mc1gt_3,146,225). coords(mc1gt_3,157,88,225).  visible(mc1gt_3,225). happensAt(walking(mc1gt_3),225).
orientation(mc1gt_2,147,226). coords(mc1gt_2,123,127,226).  visible(mc1gt_2,226). happensAt(walking(mc1gt_2),226). orientation(mc1gt_3,146,226). coords(mc1gt_3,160,88,226).  visible(mc1gt_3,226). happensAt(walking(mc1gt_3),226).
orientation(mc1gt_2,147,227). coords(mc1gt_2,123,127,227).  visible(mc1gt_2,227). happensAt(walking(mc1gt_2),227). orientation(mc1gt_3,146,227). coords(mc1gt_3,163,88,227).  visible(mc1gt_3,227). happensAt(walking(mc1gt_3),227).
orientation(mc1gt_2,147,228). coords(mc1gt_2,123,129,228).  visible(mc1gt_2,228). happensAt(walking(mc1gt_2),228). orientation(mc1gt_3,146,228). coords(mc1gt_3,165,88,228).  visible(mc1gt_3,228). happensAt(walking(mc1gt_3),228).
#show holdsAt/2.