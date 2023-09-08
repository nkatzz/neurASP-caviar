
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

	orientation(fcgt_3,159,204). coords(fcgt_3,101,164,204).  visible(fcgt_3,204). happensAt(walking(fcgt_3),204). orientation(fcgt_4,0,204). coords(fcgt_4,85,205,204).  visible(fcgt_4,204). happensAt(walking(fcgt_4),204).
orientation(fcgt_3,159,205). coords(fcgt_3,101,164,205).  visible(fcgt_3,205). happensAt(walking(fcgt_3),205). orientation(fcgt_4,0,205). coords(fcgt_4,85,205,205).  visible(fcgt_4,205). happensAt(walking(fcgt_4),205).
orientation(fcgt_3,159,206). coords(fcgt_3,101,165,206).  visible(fcgt_3,206). happensAt(walking(fcgt_3),206). orientation(fcgt_4,0,206). coords(fcgt_4,85,205,206).  visible(fcgt_4,206). happensAt(walking(fcgt_4),206).
orientation(fcgt_3,159,207). coords(fcgt_3,101,165,207).  visible(fcgt_3,207). happensAt(walking(fcgt_3),207). orientation(fcgt_4,0,207). coords(fcgt_4,85,205,207).  visible(fcgt_4,207). happensAt(walking(fcgt_4),207).
orientation(fcgt_3,153,208). coords(fcgt_3,100,166,208).  visible(fcgt_3,208). happensAt(walking(fcgt_3),208). orientation(fcgt_4,174,208). coords(fcgt_4,84,205,208).  visible(fcgt_4,208). happensAt(walking(fcgt_4),208).
orientation(fcgt_3,153,209). coords(fcgt_3,100,166,209).  visible(fcgt_3,209). happensAt(walking(fcgt_3),209). orientation(fcgt_4,172,209). coords(fcgt_4,84,205,209).  visible(fcgt_4,209). happensAt(walking(fcgt_4),209).
orientation(fcgt_3,149,210). coords(fcgt_3,100,166,210).  visible(fcgt_3,210). happensAt(walking(fcgt_3),210). orientation(fcgt_4,172,210). coords(fcgt_4,84,205,210).  visible(fcgt_4,210). happensAt(walking(fcgt_4),210).
orientation(fcgt_3,149,211). coords(fcgt_3,100,167,211).  visible(fcgt_3,211). happensAt(walking(fcgt_3),211). orientation(fcgt_4,172,211). coords(fcgt_4,84,205,211).  visible(fcgt_4,211). happensAt(walking(fcgt_4),211).
orientation(fcgt_3,149,212). coords(fcgt_3,100,167,212).  visible(fcgt_3,212). happensAt(walking(fcgt_3),212). orientation(fcgt_4,172,212). coords(fcgt_4,84,205,212).  visible(fcgt_4,212). happensAt(walking(fcgt_4),212).
orientation(fcgt_3,149,213). coords(fcgt_3,100,168,213).  visible(fcgt_3,213). happensAt(walking(fcgt_3),213). orientation(fcgt_4,172,213). coords(fcgt_4,83,205,213).  visible(fcgt_4,213). happensAt(walking(fcgt_4),213).
orientation(fcgt_3,149,214). coords(fcgt_3,100,168,214).  visible(fcgt_3,214). happensAt(walking(fcgt_3),214). orientation(fcgt_4,172,214). coords(fcgt_4,83,205,214).  visible(fcgt_4,214). happensAt(walking(fcgt_4),214).
orientation(fcgt_3,149,215). coords(fcgt_3,100,168,215).  visible(fcgt_3,215). happensAt(walking(fcgt_3),215). orientation(fcgt_4,172,215). coords(fcgt_4,82,205,215).  visible(fcgt_4,215). happensAt(walking(fcgt_4),215).
orientation(fcgt_3,149,216). coords(fcgt_3,100,168,216).  visible(fcgt_3,216). happensAt(walking(fcgt_3),216). orientation(fcgt_4,172,216). coords(fcgt_4,82,205,216).  visible(fcgt_4,216). happensAt(walking(fcgt_4),216).
orientation(fcgt_3,149,217). coords(fcgt_3,100,169,217).  visible(fcgt_3,217). happensAt(walking(fcgt_3),217). orientation(fcgt_4,172,217). coords(fcgt_4,82,205,217).  visible(fcgt_4,217). happensAt(walking(fcgt_4),217).
orientation(fcgt_3,149,218). coords(fcgt_3,100,169,218).  visible(fcgt_3,218). happensAt(walking(fcgt_3),218). orientation(fcgt_4,172,218). coords(fcgt_4,82,205,218).  visible(fcgt_4,218). happensAt(walking(fcgt_4),218).
orientation(fcgt_3,149,219). coords(fcgt_3,100,169,219).  visible(fcgt_3,219). happensAt(walking(fcgt_3),219). orientation(fcgt_4,172,219). coords(fcgt_4,81,205,219).  visible(fcgt_4,219). happensAt(walking(fcgt_4),219).
orientation(fcgt_3,149,220). coords(fcgt_3,100,170,220).  visible(fcgt_3,220). happensAt(walking(fcgt_3),220). orientation(fcgt_4,172,220). coords(fcgt_4,81,205,220).  visible(fcgt_4,220). happensAt(walking(fcgt_4),220).
orientation(fcgt_3,149,221). coords(fcgt_3,100,170,221).  visible(fcgt_3,221). happensAt(walking(fcgt_3),221). orientation(fcgt_4,172,221). coords(fcgt_4,81,205,221).  visible(fcgt_4,221). happensAt(walking(fcgt_4),221).
orientation(fcgt_3,149,222). coords(fcgt_3,100,170,222).  visible(fcgt_3,222). happensAt(walking(fcgt_3),222). orientation(fcgt_4,172,222). coords(fcgt_4,81,205,222).  visible(fcgt_4,222). happensAt(walking(fcgt_4),222).
orientation(fcgt_3,149,223). coords(fcgt_3,100,170,223).  visible(fcgt_3,223). happensAt(walking(fcgt_3),223). orientation(fcgt_4,166,223). coords(fcgt_4,81,205,223).  visible(fcgt_4,223). happensAt(inactive(fcgt_4),223).
orientation(fcgt_3,149,224). coords(fcgt_3,100,170,224).  visible(fcgt_3,224). happensAt(walking(fcgt_3),224). orientation(fcgt_4,166,224). coords(fcgt_4,81,205,224).  visible(fcgt_4,224). happensAt(inactive(fcgt_4),224).
orientation(fcgt_3,149,225). coords(fcgt_3,100,171,225).  visible(fcgt_3,225). happensAt(walking(fcgt_3),225). orientation(fcgt_4,166,225). coords(fcgt_4,80,205,225).  visible(fcgt_4,225). happensAt(inactive(fcgt_4),225).
orientation(fcgt_3,149,226). coords(fcgt_3,100,171,226).  visible(fcgt_3,226). happensAt(walking(fcgt_3),226). orientation(fcgt_4,166,226). coords(fcgt_4,80,205,226).  visible(fcgt_4,226). happensAt(inactive(fcgt_4),226).
orientation(fcgt_3,157,227). coords(fcgt_3,100,171,227).  visible(fcgt_3,227). happensAt(walking(fcgt_3),227). orientation(fcgt_4,166,227). coords(fcgt_4,80,205,227).  visible(fcgt_4,227). happensAt(inactive(fcgt_4),227).
orientation(fcgt_3,159,228). coords(fcgt_3,100,171,228).  visible(fcgt_3,228). happensAt(walking(fcgt_3),228). orientation(fcgt_4,166,228). coords(fcgt_4,80,205,228).  visible(fcgt_4,228). happensAt(inactive(fcgt_4),228).
#show holdsAt/2.