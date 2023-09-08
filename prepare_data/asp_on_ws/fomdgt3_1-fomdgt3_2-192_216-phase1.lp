
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

	orientation(fomdgt3_1,150,192). coords(fomdgt3_1,82,102,192).  visible(fomdgt3_1,192). happensAt(walking(fomdgt3_1),192). orientation(fomdgt3_2,130,192). coords(fomdgt3_2,237,153,192).  visible(fomdgt3_2,192). happensAt(walking(fomdgt3_2),192).
orientation(fomdgt3_1,150,193). coords(fomdgt3_1,82,102,193).  visible(fomdgt3_1,193). happensAt(walking(fomdgt3_1),193). orientation(fomdgt3_2,130,193). coords(fomdgt3_2,236,153,193).  visible(fomdgt3_2,193). happensAt(walking(fomdgt3_2),193).
orientation(fomdgt3_1,150,194). coords(fomdgt3_1,82,102,194).  visible(fomdgt3_1,194). happensAt(walking(fomdgt3_1),194). orientation(fomdgt3_2,130,194). coords(fomdgt3_2,235,150,194).  visible(fomdgt3_2,194). happensAt(walking(fomdgt3_2),194).
orientation(fomdgt3_1,150,195). coords(fomdgt3_1,79,99,195).  visible(fomdgt3_1,195). happensAt(walking(fomdgt3_1),195). orientation(fomdgt3_2,130,195). coords(fomdgt3_2,235,149,195).  visible(fomdgt3_2,195). happensAt(walking(fomdgt3_2),195).
orientation(fomdgt3_1,150,196). coords(fomdgt3_1,79,99,196).  visible(fomdgt3_1,196). happensAt(walking(fomdgt3_1),196). orientation(fomdgt3_2,130,196). coords(fomdgt3_2,234,148,196).  visible(fomdgt3_2,196). happensAt(walking(fomdgt3_2),196).
orientation(fomdgt3_1,150,197). coords(fomdgt3_1,79,99,197).  visible(fomdgt3_1,197). happensAt(walking(fomdgt3_1),197). orientation(fomdgt3_2,130,197). coords(fomdgt3_2,234,148,197).  visible(fomdgt3_2,197). happensAt(walking(fomdgt3_2),197).
orientation(fomdgt3_1,150,198). coords(fomdgt3_1,79,99,198).  visible(fomdgt3_1,198). happensAt(walking(fomdgt3_1),198). orientation(fomdgt3_2,130,198). coords(fomdgt3_2,234,148,198).  visible(fomdgt3_2,198). happensAt(walking(fomdgt3_2),198).
orientation(fomdgt3_1,150,199). coords(fomdgt3_1,79,99,199).  visible(fomdgt3_1,199). happensAt(walking(fomdgt3_1),199). orientation(fomdgt3_2,130,199). coords(fomdgt3_2,234,148,199).  visible(fomdgt3_2,199). happensAt(walking(fomdgt3_2),199).
orientation(fomdgt3_1,150,200). coords(fomdgt3_1,77,99,200).  visible(fomdgt3_1,200). happensAt(walking(fomdgt3_1),200). orientation(fomdgt3_2,130,200). coords(fomdgt3_2,233,145,200).  visible(fomdgt3_2,200). happensAt(walking(fomdgt3_2),200).
orientation(fomdgt3_1,150,201). coords(fomdgt3_1,77,98,201).  visible(fomdgt3_1,201). happensAt(walking(fomdgt3_1),201). orientation(fomdgt3_2,130,201). coords(fomdgt3_2,232,144,201).  visible(fomdgt3_2,201). happensAt(walking(fomdgt3_2),201).
orientation(fomdgt3_1,150,202). coords(fomdgt3_1,77,98,202).  visible(fomdgt3_1,202). happensAt(walking(fomdgt3_1),202). orientation(fomdgt3_2,130,202). coords(fomdgt3_2,232,143,202).  visible(fomdgt3_2,202). happensAt(walking(fomdgt3_2),202).
orientation(fomdgt3_1,150,203). coords(fomdgt3_1,77,98,203).  visible(fomdgt3_1,203). happensAt(walking(fomdgt3_1),203). orientation(fomdgt3_2,130,203). coords(fomdgt3_2,232,142,203).  visible(fomdgt3_2,203). happensAt(walking(fomdgt3_2),203).
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
#show holdsAt/2.