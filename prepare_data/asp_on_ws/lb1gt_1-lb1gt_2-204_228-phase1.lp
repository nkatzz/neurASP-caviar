
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

	orientation(lb1gt_1,151,204). coords(lb1gt_1,121,158,204).  visible(lb1gt_1,204). happensAt(walking(lb1gt_1),204). orientation(lb1gt_2,147,204). coords(lb1gt_2,130,116,204).  visible(lb1gt_2,204). happensAt(walking(lb1gt_2),204).
orientation(lb1gt_1,151,205). coords(lb1gt_1,121,158,205).  visible(lb1gt_1,205). happensAt(walking(lb1gt_1),205). orientation(lb1gt_2,147,205). coords(lb1gt_2,131,116,205).  visible(lb1gt_2,205). happensAt(walking(lb1gt_2),205).
orientation(lb1gt_1,151,206). coords(lb1gt_1,121,158,206).  visible(lb1gt_1,206). happensAt(walking(lb1gt_1),206). orientation(lb1gt_2,147,206). coords(lb1gt_2,131,116,206).  visible(lb1gt_2,206). happensAt(walking(lb1gt_2),206).
orientation(lb1gt_1,151,207). coords(lb1gt_1,121,158,207).  visible(lb1gt_1,207). happensAt(walking(lb1gt_1),207). orientation(lb1gt_2,147,207). coords(lb1gt_2,132,116,207).  visible(lb1gt_2,207). happensAt(walking(lb1gt_2),207).
orientation(lb1gt_1,151,208). coords(lb1gt_1,121,158,208).  visible(lb1gt_1,208). happensAt(walking(lb1gt_1),208). orientation(lb1gt_2,147,208). coords(lb1gt_2,132,116,208).  visible(lb1gt_2,208). happensAt(walking(lb1gt_2),208).
orientation(lb1gt_1,151,209). coords(lb1gt_1,121,159,209).  visible(lb1gt_1,209). happensAt(walking(lb1gt_1),209). orientation(lb1gt_2,147,209). coords(lb1gt_2,132,116,209).  visible(lb1gt_2,209). happensAt(walking(lb1gt_2),209).
orientation(lb1gt_1,151,210). coords(lb1gt_1,122,159,210).  visible(lb1gt_1,210). happensAt(walking(lb1gt_1),210). orientation(lb1gt_2,147,210). coords(lb1gt_2,133,117,210).  visible(lb1gt_2,210). happensAt(walking(lb1gt_2),210).
orientation(lb1gt_1,151,211). coords(lb1gt_1,122,159,211).  visible(lb1gt_1,211). happensAt(walking(lb1gt_1),211). orientation(lb1gt_2,147,211). coords(lb1gt_2,133,117,211).  visible(lb1gt_2,211). happensAt(walking(lb1gt_2),211).
orientation(lb1gt_1,151,212). coords(lb1gt_1,123,159,212).  visible(lb1gt_1,212). happensAt(walking(lb1gt_1),212). orientation(lb1gt_2,147,212). coords(lb1gt_2,133,118,212).  visible(lb1gt_2,212). happensAt(walking(lb1gt_2),212).
orientation(lb1gt_1,151,213). coords(lb1gt_1,124,159,213).  visible(lb1gt_1,213). happensAt(walking(lb1gt_1),213). orientation(lb1gt_2,147,213). coords(lb1gt_2,134,119,213).  visible(lb1gt_2,213). happensAt(walking(lb1gt_2),213).
orientation(lb1gt_1,151,214). coords(lb1gt_1,126,159,214).  visible(lb1gt_1,214). happensAt(walking(lb1gt_1),214). orientation(lb1gt_2,147,214). coords(lb1gt_2,135,120,214).  visible(lb1gt_2,214). happensAt(walking(lb1gt_2),214).
orientation(lb1gt_1,151,215). coords(lb1gt_1,127,159,215).  visible(lb1gt_1,215). happensAt(walking(lb1gt_1),215). orientation(lb1gt_2,147,215). coords(lb1gt_2,135,120,215).  visible(lb1gt_2,215). happensAt(walking(lb1gt_2),215).
orientation(lb1gt_1,151,216). coords(lb1gt_1,129,159,216).  visible(lb1gt_1,216). happensAt(walking(lb1gt_1),216). orientation(lb1gt_2,147,216). coords(lb1gt_2,136,120,216).  visible(lb1gt_2,216). happensAt(walking(lb1gt_2),216).
orientation(lb1gt_1,151,217). coords(lb1gt_1,130,159,217).  visible(lb1gt_1,217). happensAt(walking(lb1gt_1),217). orientation(lb1gt_2,147,217). coords(lb1gt_2,137,121,217).  visible(lb1gt_2,217). happensAt(walking(lb1gt_2),217).
orientation(lb1gt_1,151,218). coords(lb1gt_1,130,159,218).  visible(lb1gt_1,218). happensAt(walking(lb1gt_1),218). orientation(lb1gt_2,147,218). coords(lb1gt_2,137,121,218).  visible(lb1gt_2,218). happensAt(walking(lb1gt_2),218).
orientation(lb1gt_1,151,219). coords(lb1gt_1,131,159,219).  visible(lb1gt_1,219). happensAt(walking(lb1gt_1),219). orientation(lb1gt_2,147,219). coords(lb1gt_2,138,121,219).  visible(lb1gt_2,219). happensAt(walking(lb1gt_2),219).
orientation(lb1gt_1,151,220). coords(lb1gt_1,132,159,220).  visible(lb1gt_1,220). happensAt(walking(lb1gt_1),220). orientation(lb1gt_2,147,220). coords(lb1gt_2,138,122,220).  visible(lb1gt_2,220). happensAt(walking(lb1gt_2),220).
orientation(lb1gt_1,151,221). coords(lb1gt_1,133,159,221).  visible(lb1gt_1,221). happensAt(walking(lb1gt_1),221). orientation(lb1gt_2,147,221). coords(lb1gt_2,138,122,221).  visible(lb1gt_2,221). happensAt(walking(lb1gt_2),221).
orientation(lb1gt_1,151,222). coords(lb1gt_1,133,159,222).  visible(lb1gt_1,222). happensAt(walking(lb1gt_1),222). orientation(lb1gt_2,147,222). coords(lb1gt_2,138,122,222).  visible(lb1gt_2,222). happensAt(walking(lb1gt_2),222).
orientation(lb1gt_1,151,223). coords(lb1gt_1,134,159,223).  visible(lb1gt_1,223). happensAt(walking(lb1gt_1),223). orientation(lb1gt_2,147,223). coords(lb1gt_2,138,122,223).  visible(lb1gt_2,223). happensAt(walking(lb1gt_2),223).
orientation(lb1gt_1,151,224). coords(lb1gt_1,134,158,224).  visible(lb1gt_1,224). happensAt(walking(lb1gt_1),224). orientation(lb1gt_2,147,224). coords(lb1gt_2,138,122,224).  visible(lb1gt_2,224). happensAt(walking(lb1gt_2),224).
orientation(lb1gt_1,151,225). coords(lb1gt_1,134,158,225).  visible(lb1gt_1,225). happensAt(walking(lb1gt_1),225). orientation(lb1gt_2,147,225). coords(lb1gt_2,138,122,225).  visible(lb1gt_2,225). happensAt(walking(lb1gt_2),225).
orientation(lb1gt_1,151,226). coords(lb1gt_1,135,158,226).  visible(lb1gt_1,226). happensAt(walking(lb1gt_1),226). orientation(lb1gt_2,147,226). coords(lb1gt_2,139,122,226).  visible(lb1gt_2,226). happensAt(walking(lb1gt_2),226).
orientation(lb1gt_1,151,227). coords(lb1gt_1,135,158,227).  visible(lb1gt_1,227). happensAt(walking(lb1gt_1),227). orientation(lb1gt_2,147,227). coords(lb1gt_2,141,123,227).  visible(lb1gt_2,227). happensAt(walking(lb1gt_2),227).
orientation(lb1gt_1,151,228). coords(lb1gt_1,135,158,228).  visible(lb1gt_1,228). happensAt(walking(lb1gt_1),228). orientation(lb1gt_2,147,228). coords(lb1gt_2,142,123,228).  visible(lb1gt_2,228). happensAt(walking(lb1gt_2),228).
#show holdsAt/2.