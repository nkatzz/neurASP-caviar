
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

	orientation(fomdgt3_1,150,204). coords(fomdgt3_1,77,98,204).  visible(fomdgt3_1,204). happensAt(walking(fomdgt3_1),204). orientation(fomdgt3_2,130,204). coords(fomdgt3_2,232,142,204).  visible(fomdgt3_2,204). happensAt(walking(fomdgt3_2),204).
orientation(fomdgt3_1,150,205). coords(fomdgt3_1,77,98,205).  visible(fomdgt3_1,205). happensAt(walking(fomdgt3_1),205). orientation(fomdgt3_2,130,205). coords(fomdgt3_2,231,141,205).  visible(fomdgt3_2,205). happensAt(walking(fomdgt3_2),205).
orientation(fomdgt3_1,150,206). coords(fomdgt3_1,77,98,206).  visible(fomdgt3_1,206). happensAt(walking(fomdgt3_1),206). orientation(fomdgt3_2,130,206). coords(fomdgt3_2,231,134,206).  visible(fomdgt3_2,206). happensAt(walking(fomdgt3_2),206).
orientation(fomdgt3_1,150,207). coords(fomdgt3_1,77,98,207).  visible(fomdgt3_1,207). happensAt(walking(fomdgt3_1),207). orientation(fomdgt3_2,130,207). coords(fomdgt3_2,231,134,207).  visible(fomdgt3_2,207). happensAt(walking(fomdgt3_2),207).
orientation(fomdgt3_1,150,208). coords(fomdgt3_1,74,98,208).  visible(fomdgt3_1,208). happensAt(walking(fomdgt3_1),208). orientation(fomdgt3_2,130,208). coords(fomdgt3_2,231,134,208).  visible(fomdgt3_2,208). happensAt(walking(fomdgt3_2),208).
orientation(fomdgt3_1,150,209). coords(fomdgt3_1,74,98,209).  visible(fomdgt3_1,209). happensAt(walking(fomdgt3_1),209). orientation(fomdgt3_2,130,209). coords(fomdgt3_2,231,134,209).  visible(fomdgt3_2,209). happensAt(walking(fomdgt3_2),209).
orientation(fomdgt3_1,150,210). coords(fomdgt3_1,74,98,210).  visible(fomdgt3_1,210). happensAt(walking(fomdgt3_1),210). orientation(fomdgt3_2,130,210). coords(fomdgt3_2,231,130,210).  visible(fomdgt3_2,210). happensAt(walking(fomdgt3_2),210).
orientation(fomdgt3_1,150,211). coords(fomdgt3_1,73,96,211).  visible(fomdgt3_1,211). happensAt(walking(fomdgt3_1),211). orientation(fomdgt3_2,130,211). coords(fomdgt3_2,228,130,211).  visible(fomdgt3_2,211). happensAt(walking(fomdgt3_2),211).
orientation(fomdgt3_1,150,212). coords(fomdgt3_1,73,96,212).  visible(fomdgt3_1,212). happensAt(walking(fomdgt3_1),212). orientation(fomdgt3_2,130,212). coords(fomdgt3_2,228,130,212).  visible(fomdgt3_2,212). happensAt(walking(fomdgt3_2),212).
orientation(fomdgt3_1,150,213). coords(fomdgt3_1,73,96,213).  visible(fomdgt3_1,213). happensAt(walking(fomdgt3_1),213). orientation(fomdgt3_2,130,213). coords(fomdgt3_2,228,130,213).  visible(fomdgt3_2,213). happensAt(walking(fomdgt3_2),213).
orientation(fomdgt3_1,150,214). coords(fomdgt3_1,73,96,214).  visible(fomdgt3_1,214). happensAt(walking(fomdgt3_1),214). orientation(fomdgt3_2,130,214). coords(fomdgt3_2,228,130,214).  visible(fomdgt3_2,214). happensAt(walking(fomdgt3_2),214).
orientation(fomdgt3_1,150,215). coords(fomdgt3_1,73,96,215).  visible(fomdgt3_1,215). happensAt(walking(fomdgt3_1),215). orientation(fomdgt3_2,130,215). coords(fomdgt3_2,228,130,215).  visible(fomdgt3_2,215). happensAt(walking(fomdgt3_2),215).
orientation(fomdgt3_1,150,216). coords(fomdgt3_1,73,96,216).  visible(fomdgt3_1,216). happensAt(walking(fomdgt3_1),216). orientation(fomdgt3_2,130,216). coords(fomdgt3_2,228,128,216).  visible(fomdgt3_2,216). happensAt(walking(fomdgt3_2),216).
orientation(fomdgt3_1,150,217). coords(fomdgt3_1,73,96,217).  visible(fomdgt3_1,217). happensAt(walking(fomdgt3_1),217). orientation(fomdgt3_2,130,217). coords(fomdgt3_2,228,128,217).  visible(fomdgt3_2,217). happensAt(walking(fomdgt3_2),217).
orientation(fomdgt3_1,150,218). coords(fomdgt3_1,73,96,218).  visible(fomdgt3_1,218). happensAt(walking(fomdgt3_1),218). orientation(fomdgt3_2,130,218). coords(fomdgt3_2,228,128,218).  visible(fomdgt3_2,218). happensAt(walking(fomdgt3_2),218).
orientation(fomdgt3_1,150,219). coords(fomdgt3_1,73,96,219).  visible(fomdgt3_1,219). happensAt(walking(fomdgt3_1),219). orientation(fomdgt3_2,130,219). coords(fomdgt3_2,228,128,219).  visible(fomdgt3_2,219). happensAt(walking(fomdgt3_2),219).
orientation(fomdgt3_1,150,220). coords(fomdgt3_1,73,96,220).  visible(fomdgt3_1,220). happensAt(walking(fomdgt3_1),220). orientation(fomdgt3_2,130,220). coords(fomdgt3_2,228,128,220).  visible(fomdgt3_2,220). happensAt(walking(fomdgt3_2),220).
orientation(fomdgt3_1,150,221). coords(fomdgt3_1,69,94,221).  visible(fomdgt3_1,221). happensAt(walking(fomdgt3_1),221). orientation(fomdgt3_2,130,221). coords(fomdgt3_2,228,128,221).  visible(fomdgt3_2,221). happensAt(walking(fomdgt3_2),221).
orientation(fomdgt3_1,150,222). coords(fomdgt3_1,69,94,222).  visible(fomdgt3_1,222). happensAt(walking(fomdgt3_1),222). orientation(fomdgt3_2,130,222). coords(fomdgt3_2,228,128,222).  visible(fomdgt3_2,222). happensAt(walking(fomdgt3_2),222).
orientation(fomdgt3_1,150,223). coords(fomdgt3_1,69,94,223).  visible(fomdgt3_1,223). happensAt(walking(fomdgt3_1),223). orientation(fomdgt3_2,130,223). coords(fomdgt3_2,228,128,223).  visible(fomdgt3_2,223). happensAt(walking(fomdgt3_2),223).
orientation(fomdgt3_1,150,224). coords(fomdgt3_1,69,94,224).  visible(fomdgt3_1,224). happensAt(walking(fomdgt3_1),224). orientation(fomdgt3_2,130,224). coords(fomdgt3_2,228,128,224).  visible(fomdgt3_2,224). happensAt(walking(fomdgt3_2),224).
orientation(fomdgt3_1,150,225). coords(fomdgt3_1,69,94,225).  visible(fomdgt3_1,225). happensAt(walking(fomdgt3_1),225). orientation(fomdgt3_2,130,225). coords(fomdgt3_2,228,128,225).  visible(fomdgt3_2,225). happensAt(walking(fomdgt3_2),225).
orientation(fomdgt3_1,150,226). coords(fomdgt3_1,68,93,226).  visible(fomdgt3_1,226). happensAt(walking(fomdgt3_1),226). orientation(fomdgt3_2,130,226). coords(fomdgt3_2,228,128,226).  visible(fomdgt3_2,226). happensAt(walking(fomdgt3_2),226).
orientation(fomdgt3_1,150,227). coords(fomdgt3_1,68,93,227).  visible(fomdgt3_1,227). happensAt(walking(fomdgt3_1),227). orientation(fomdgt3_2,130,227). coords(fomdgt3_2,228,128,227).  visible(fomdgt3_2,227). happensAt(walking(fomdgt3_2),227).
orientation(fomdgt3_1,150,228). coords(fomdgt3_1,68,93,228).  visible(fomdgt3_1,228). happensAt(walking(fomdgt3_1),228). orientation(fomdgt3_2,130,228). coords(fomdgt3_2,228,128,228).  visible(fomdgt3_2,228). happensAt(walking(fomdgt3_2),228).
#show holdsAt/2.