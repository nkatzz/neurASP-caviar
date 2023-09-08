
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

	orientation(lbgt_0,150,204). coords(lbgt_0,189,165,204).  visible(lbgt_0,204). happensAt(walking(lbgt_0),204). orientation(lbgt_1,147,204). coords(lbgt_1,80,88,204).  visible(lbgt_1,204). happensAt(walking(lbgt_1),204).
orientation(lbgt_0,156,205). coords(lbgt_0,188,165,205).  visible(lbgt_0,205). happensAt(walking(lbgt_0),205). orientation(lbgt_1,147,205). coords(lbgt_1,80,88,205).  visible(lbgt_1,205). happensAt(walking(lbgt_1),205).
orientation(lbgt_0,156,206). coords(lbgt_0,188,164,206).  visible(lbgt_0,206). happensAt(walking(lbgt_0),206). orientation(lbgt_1,147,206). coords(lbgt_1,80,88,206).  visible(lbgt_1,206). happensAt(walking(lbgt_1),206).
orientation(lbgt_0,154,207). coords(lbgt_0,187,164,207).  visible(lbgt_0,207). happensAt(walking(lbgt_0),207). orientation(lbgt_1,147,207). coords(lbgt_1,80,87,207).  visible(lbgt_1,207). happensAt(walking(lbgt_1),207).
orientation(lbgt_0,154,208). coords(lbgt_0,187,163,208).  visible(lbgt_0,208). happensAt(walking(lbgt_0),208). orientation(lbgt_1,147,208). coords(lbgt_1,80,86,208).  visible(lbgt_1,208). happensAt(walking(lbgt_1),208).
orientation(lbgt_0,151,209). coords(lbgt_0,187,162,209).  visible(lbgt_0,209). happensAt(walking(lbgt_0),209). orientation(lbgt_1,147,209). coords(lbgt_1,81,84,209).  visible(lbgt_1,209). happensAt(walking(lbgt_1),209).
orientation(lbgt_0,151,210). coords(lbgt_0,186,162,210).  visible(lbgt_0,210). happensAt(walking(lbgt_0),210). orientation(lbgt_1,147,210). coords(lbgt_1,81,83,210).  visible(lbgt_1,210). happensAt(walking(lbgt_1),210).
orientation(lbgt_0,151,211). coords(lbgt_0,184,160,211).  visible(lbgt_0,211). happensAt(walking(lbgt_0),211). orientation(lbgt_1,147,211). coords(lbgt_1,81,82,211).  visible(lbgt_1,211). happensAt(walking(lbgt_1),211).
orientation(lbgt_0,151,212). coords(lbgt_0,183,159,212).  visible(lbgt_0,212). happensAt(walking(lbgt_0),212). orientation(lbgt_1,147,212). coords(lbgt_1,81,82,212).  visible(lbgt_1,212). happensAt(walking(lbgt_1),212).
orientation(lbgt_0,147,213). coords(lbgt_0,181,157,213).  visible(lbgt_0,213). happensAt(walking(lbgt_0),213). orientation(lbgt_1,147,213). coords(lbgt_1,81,82,213).  visible(lbgt_1,213). happensAt(walking(lbgt_1),213).
orientation(lbgt_0,147,214). coords(lbgt_0,179,155,214).  visible(lbgt_0,214). happensAt(walking(lbgt_0),214). orientation(lbgt_1,147,214). coords(lbgt_1,81,82,214).  visible(lbgt_1,214). happensAt(walking(lbgt_1),214).
orientation(lbgt_0,147,215). coords(lbgt_0,177,153,215).  visible(lbgt_0,215). happensAt(walking(lbgt_0),215). orientation(lbgt_1,147,215). coords(lbgt_1,81,81,215).  visible(lbgt_1,215). happensAt(walking(lbgt_1),215).
orientation(lbgt_0,147,216). coords(lbgt_0,175,152,216).  visible(lbgt_0,216). happensAt(walking(lbgt_0),216). orientation(lbgt_1,147,216). coords(lbgt_1,81,81,216).  visible(lbgt_1,216). happensAt(walking(lbgt_1),216).
orientation(lbgt_0,150,217). coords(lbgt_0,174,148,217).  visible(lbgt_0,217). happensAt(walking(lbgt_0),217). orientation(lbgt_1,147,217). coords(lbgt_1,81,80,217).  visible(lbgt_1,217). happensAt(walking(lbgt_1),217).
orientation(lbgt_0,150,218). coords(lbgt_0,173,146,218).  visible(lbgt_0,218). happensAt(walking(lbgt_0),218). orientation(lbgt_1,147,218). coords(lbgt_1,81,79,218).  visible(lbgt_1,218). happensAt(walking(lbgt_1),218).
orientation(lbgt_0,150,219). coords(lbgt_0,173,144,219).  visible(lbgt_0,219). happensAt(walking(lbgt_0),219). orientation(lbgt_1,147,219). coords(lbgt_1,81,79,219).  visible(lbgt_1,219). happensAt(walking(lbgt_1),219).
orientation(lbgt_0,155,220). coords(lbgt_0,173,144,220).  visible(lbgt_0,220). happensAt(walking(lbgt_0),220). orientation(lbgt_1,147,220). coords(lbgt_1,81,77,220).  visible(lbgt_1,220). happensAt(walking(lbgt_1),220).
orientation(lbgt_0,155,221). coords(lbgt_0,172,144,221).  visible(lbgt_0,221). happensAt(walking(lbgt_0),221). orientation(lbgt_1,147,221). coords(lbgt_1,81,76,221).  visible(lbgt_1,221). happensAt(walking(lbgt_1),221).
orientation(lbgt_0,155,222). coords(lbgt_0,172,144,222).  visible(lbgt_0,222). happensAt(walking(lbgt_0),222). orientation(lbgt_1,147,222). coords(lbgt_1,81,76,222).  visible(lbgt_1,222). happensAt(walking(lbgt_1),222).
orientation(lbgt_0,155,223). coords(lbgt_0,172,144,223).  visible(lbgt_0,223). happensAt(walking(lbgt_0),223). orientation(lbgt_1,147,223). coords(lbgt_1,81,75,223).  visible(lbgt_1,223). happensAt(walking(lbgt_1),223).
orientation(lbgt_0,155,224). coords(lbgt_0,171,144,224).  visible(lbgt_0,224). happensAt(walking(lbgt_0),224). orientation(lbgt_1,147,224). coords(lbgt_1,81,74,224).  visible(lbgt_1,224). happensAt(walking(lbgt_1),224).
orientation(lbgt_0,155,225). coords(lbgt_0,170,143,225).  visible(lbgt_0,225). happensAt(walking(lbgt_0),225). orientation(lbgt_1,147,225). coords(lbgt_1,81,74,225).  visible(lbgt_1,225). happensAt(walking(lbgt_1),225).
orientation(lbgt_0,155,226). coords(lbgt_0,169,142,226).  visible(lbgt_0,226). happensAt(walking(lbgt_0),226). orientation(lbgt_1,147,226). coords(lbgt_1,81,73,226).  visible(lbgt_1,226). happensAt(walking(lbgt_1),226).
orientation(lbgt_0,155,227). coords(lbgt_0,168,140,227).  visible(lbgt_0,227). happensAt(walking(lbgt_0),227). orientation(lbgt_1,147,227). coords(lbgt_1,81,73,227).  visible(lbgt_1,227). happensAt(walking(lbgt_1),227).
orientation(lbgt_0,158,228). coords(lbgt_0,167,139,228).  visible(lbgt_0,228). happensAt(walking(lbgt_0),228). orientation(lbgt_1,147,228). coords(lbgt_1,81,72,228).  visible(lbgt_1,228). happensAt(walking(lbgt_1),228).
#show holdsAt/2.