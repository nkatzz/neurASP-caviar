
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

	orientation(mc1gt_1,137,204). coords(mc1gt_1,161,109,204).  visible(mc1gt_1,204). happensAt(walking(mc1gt_1),204). orientation(mc1gt_3,146,204). coords(mc1gt_3,133,84,204).  visible(mc1gt_3,204). happensAt(walking(mc1gt_3),204).
orientation(mc1gt_1,137,205). coords(mc1gt_1,162,109,205).  visible(mc1gt_1,205). happensAt(walking(mc1gt_1),205). orientation(mc1gt_3,146,205). coords(mc1gt_3,134,84,205).  visible(mc1gt_3,205). happensAt(walking(mc1gt_3),205).
orientation(mc1gt_1,140,206). coords(mc1gt_1,162,110,206).  visible(mc1gt_1,206). happensAt(walking(mc1gt_1),206). orientation(mc1gt_3,146,206). coords(mc1gt_3,136,84,206).  visible(mc1gt_3,206). happensAt(walking(mc1gt_3),206).
orientation(mc1gt_1,140,207). coords(mc1gt_1,162,110,207).  visible(mc1gt_1,207). happensAt(walking(mc1gt_1),207). orientation(mc1gt_3,146,207). coords(mc1gt_3,137,82,207).  visible(mc1gt_3,207). happensAt(walking(mc1gt_3),207).
orientation(mc1gt_1,140,208). coords(mc1gt_1,162,110,208).  visible(mc1gt_1,208). happensAt(walking(mc1gt_1),208). orientation(mc1gt_3,146,208). coords(mc1gt_3,138,82,208).  visible(mc1gt_3,208). happensAt(walking(mc1gt_3),208).
orientation(mc1gt_1,141,209). coords(mc1gt_1,163,111,209).  visible(mc1gt_1,209). happensAt(walking(mc1gt_1),209). orientation(mc1gt_3,146,209). coords(mc1gt_3,138,83,209).  visible(mc1gt_3,209). happensAt(walking(mc1gt_3),209).
orientation(mc1gt_1,141,210). coords(mc1gt_1,163,111,210).  visible(mc1gt_1,210). happensAt(walking(mc1gt_1),210). orientation(mc1gt_3,146,210). coords(mc1gt_3,139,83,210).  visible(mc1gt_3,210). happensAt(walking(mc1gt_3),210).
orientation(mc1gt_1,143,211). coords(mc1gt_1,163,111,211).  visible(mc1gt_1,211). happensAt(walking(mc1gt_1),211). orientation(mc1gt_3,146,211). coords(mc1gt_3,141,83,211).  visible(mc1gt_3,211). happensAt(walking(mc1gt_3),211).
orientation(mc1gt_1,142,212). coords(mc1gt_1,164,112,212).  visible(mc1gt_1,212). happensAt(walking(mc1gt_1),212). orientation(mc1gt_3,146,212). coords(mc1gt_3,142,83,212).  visible(mc1gt_3,212). happensAt(walking(mc1gt_3),212).
orientation(mc1gt_1,143,213). coords(mc1gt_1,167,112,213).  visible(mc1gt_1,213). happensAt(walking(mc1gt_1),213). orientation(mc1gt_3,146,213). coords(mc1gt_3,142,83,213).  visible(mc1gt_3,213). happensAt(walking(mc1gt_3),213).
orientation(mc1gt_1,143,214). coords(mc1gt_1,169,113,214).  visible(mc1gt_1,214). happensAt(walking(mc1gt_1),214). orientation(mc1gt_3,146,214). coords(mc1gt_3,143,83,214).  visible(mc1gt_3,214). happensAt(walking(mc1gt_3),214).
orientation(mc1gt_1,143,215). coords(mc1gt_1,171,114,215).  visible(mc1gt_1,215). happensAt(walking(mc1gt_1),215). orientation(mc1gt_3,146,215). coords(mc1gt_3,144,84,215).  visible(mc1gt_3,215). happensAt(walking(mc1gt_3),215).
orientation(mc1gt_1,142,216). coords(mc1gt_1,173,115,216).  visible(mc1gt_1,216). happensAt(walking(mc1gt_1),216). orientation(mc1gt_3,146,216). coords(mc1gt_3,145,84,216).  visible(mc1gt_3,216). happensAt(walking(mc1gt_3),216).
orientation(mc1gt_1,141,217). coords(mc1gt_1,174,116,217).  visible(mc1gt_1,217). happensAt(walking(mc1gt_1),217). orientation(mc1gt_3,146,217). coords(mc1gt_3,148,85,217).  visible(mc1gt_3,217). happensAt(walking(mc1gt_3),217).
orientation(mc1gt_1,141,218). coords(mc1gt_1,175,117,218).  visible(mc1gt_1,218). happensAt(walking(mc1gt_1),218). orientation(mc1gt_3,146,218). coords(mc1gt_3,151,86,218).  visible(mc1gt_3,218). happensAt(walking(mc1gt_3),218).
orientation(mc1gt_1,141,219). coords(mc1gt_1,176,118,219).  visible(mc1gt_1,219). happensAt(walking(mc1gt_1),219). orientation(mc1gt_3,146,219). coords(mc1gt_3,153,87,219).  visible(mc1gt_3,219). happensAt(walking(mc1gt_3),219).
orientation(mc1gt_1,141,220). coords(mc1gt_1,177,119,220).  visible(mc1gt_1,220). happensAt(walking(mc1gt_1),220). orientation(mc1gt_3,146,220). coords(mc1gt_3,154,87,220).  visible(mc1gt_3,220). happensAt(walking(mc1gt_3),220).
orientation(mc1gt_1,139,221). coords(mc1gt_1,178,119,221).  visible(mc1gt_1,221). happensAt(walking(mc1gt_1),221). orientation(mc1gt_3,146,221). coords(mc1gt_3,154,87,221).  visible(mc1gt_3,221). happensAt(walking(mc1gt_3),221).
orientation(mc1gt_1,138,222). coords(mc1gt_1,178,120,222).  visible(mc1gt_1,222). happensAt(walking(mc1gt_1),222). orientation(mc1gt_3,146,222). coords(mc1gt_3,154,87,222).  visible(mc1gt_3,222). happensAt(walking(mc1gt_3),222).
orientation(mc1gt_1,138,223). coords(mc1gt_1,178,120,223).  visible(mc1gt_1,223). happensAt(walking(mc1gt_1),223). orientation(mc1gt_3,146,223). coords(mc1gt_3,155,88,223).  visible(mc1gt_3,223). happensAt(walking(mc1gt_3),223).
orientation(mc1gt_1,138,224). coords(mc1gt_1,179,120,224).  visible(mc1gt_1,224). happensAt(walking(mc1gt_1),224). orientation(mc1gt_3,146,224). coords(mc1gt_3,155,88,224).  visible(mc1gt_3,224). happensAt(walking(mc1gt_3),224).
orientation(mc1gt_1,139,225). coords(mc1gt_1,179,121,225).  visible(mc1gt_1,225). happensAt(walking(mc1gt_1),225). orientation(mc1gt_3,146,225). coords(mc1gt_3,157,88,225).  visible(mc1gt_3,225). happensAt(walking(mc1gt_3),225).
orientation(mc1gt_1,142,226). coords(mc1gt_1,179,121,226).  visible(mc1gt_1,226). happensAt(walking(mc1gt_1),226). orientation(mc1gt_3,146,226). coords(mc1gt_3,160,88,226).  visible(mc1gt_3,226). happensAt(walking(mc1gt_3),226).
orientation(mc1gt_1,142,227). coords(mc1gt_1,179,123,227).  visible(mc1gt_1,227). happensAt(walking(mc1gt_1),227). orientation(mc1gt_3,146,227). coords(mc1gt_3,163,88,227).  visible(mc1gt_3,227). happensAt(walking(mc1gt_3),227).
orientation(mc1gt_1,142,228). coords(mc1gt_1,179,126,228).  visible(mc1gt_1,228). happensAt(walking(mc1gt_1),228). orientation(mc1gt_3,146,228). coords(mc1gt_3,165,88,228).  visible(mc1gt_3,228). happensAt(walking(mc1gt_3),228).
#show holdsAt/2.