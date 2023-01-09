
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

	orientation(fomdgt3_1,150,186). coords(fomdgt3_1,82,102,186).  visible(fomdgt3_1,186). happensAt(walking(fomdgt3_1),186). orientation(fomdgt3_6,130,186). coords(fomdgt3_6,220,63,186).  visible(fomdgt3_6,186). happensAt(walking(fomdgt3_6),186).
orientation(fomdgt3_1,150,187). coords(fomdgt3_1,82,102,187).  visible(fomdgt3_1,187). happensAt(walking(fomdgt3_1),187). orientation(fomdgt3_6,130,187). coords(fomdgt3_6,220,64,187).  visible(fomdgt3_6,187). happensAt(walking(fomdgt3_6),187).
orientation(fomdgt3_1,150,188). coords(fomdgt3_1,82,102,188).  visible(fomdgt3_1,188). happensAt(walking(fomdgt3_1),188). orientation(fomdgt3_6,130,188). coords(fomdgt3_6,220,66,188).  visible(fomdgt3_6,188). happensAt(walking(fomdgt3_6),188).
orientation(fomdgt3_1,150,189). coords(fomdgt3_1,82,102,189).  visible(fomdgt3_1,189). happensAt(walking(fomdgt3_1),189). orientation(fomdgt3_6,130,189). coords(fomdgt3_6,220,68,189).  visible(fomdgt3_6,189). happensAt(walking(fomdgt3_6),189).
orientation(fomdgt3_1,150,190). coords(fomdgt3_1,82,102,190).  visible(fomdgt3_1,190). happensAt(walking(fomdgt3_1),190). orientation(fomdgt3_6,130,190). coords(fomdgt3_6,219,69,190).  visible(fomdgt3_6,190). happensAt(walking(fomdgt3_6),190).
orientation(fomdgt3_1,150,191). coords(fomdgt3_1,82,102,191).  visible(fomdgt3_1,191). happensAt(walking(fomdgt3_1),191). orientation(fomdgt3_6,130,191). coords(fomdgt3_6,218,70,191).  visible(fomdgt3_6,191). happensAt(walking(fomdgt3_6),191).
orientation(fomdgt3_1,150,192). coords(fomdgt3_1,82,102,192).  visible(fomdgt3_1,192). happensAt(walking(fomdgt3_1),192). orientation(fomdgt3_6,130,192). coords(fomdgt3_6,218,71,192).  visible(fomdgt3_6,192). happensAt(walking(fomdgt3_6),192).
orientation(fomdgt3_1,150,193). coords(fomdgt3_1,82,102,193).  visible(fomdgt3_1,193). happensAt(walking(fomdgt3_1),193). orientation(fomdgt3_6,130,193). coords(fomdgt3_6,218,70,193).  visible(fomdgt3_6,193). happensAt(walking(fomdgt3_6),193).
orientation(fomdgt3_1,150,194). coords(fomdgt3_1,82,102,194).  visible(fomdgt3_1,194). happensAt(walking(fomdgt3_1),194). orientation(fomdgt3_6,130,194). coords(fomdgt3_6,217,72,194).  visible(fomdgt3_6,194). happensAt(walking(fomdgt3_6),194).
orientation(fomdgt3_1,150,195). coords(fomdgt3_1,79,99,195).  visible(fomdgt3_1,195). happensAt(walking(fomdgt3_1),195). orientation(fomdgt3_6,130,195). coords(fomdgt3_6,216,72,195).  visible(fomdgt3_6,195). happensAt(walking(fomdgt3_6),195).
orientation(fomdgt3_1,150,196). coords(fomdgt3_1,79,99,196).  visible(fomdgt3_1,196). happensAt(walking(fomdgt3_1),196). orientation(fomdgt3_6,130,196). coords(fomdgt3_6,215,72,196).  visible(fomdgt3_6,196). happensAt(walking(fomdgt3_6),196).
orientation(fomdgt3_1,150,197). coords(fomdgt3_1,79,99,197).  visible(fomdgt3_1,197). happensAt(walking(fomdgt3_1),197). orientation(fomdgt3_6,130,197). coords(fomdgt3_6,212,75,197).  visible(fomdgt3_6,197). happensAt(walking(fomdgt3_6),197).
orientation(fomdgt3_1,150,198). coords(fomdgt3_1,79,99,198).  visible(fomdgt3_1,198). happensAt(walking(fomdgt3_1),198). orientation(fomdgt3_6,130,198). coords(fomdgt3_6,211,79,198).  visible(fomdgt3_6,198). happensAt(walking(fomdgt3_6),198).
orientation(fomdgt3_1,150,199). coords(fomdgt3_1,79,99,199).  visible(fomdgt3_1,199). happensAt(walking(fomdgt3_1),199). orientation(fomdgt3_6,130,199). coords(fomdgt3_6,211,79,199).  visible(fomdgt3_6,199). happensAt(walking(fomdgt3_6),199).
orientation(fomdgt3_1,150,200). coords(fomdgt3_1,77,99,200).  visible(fomdgt3_1,200). happensAt(walking(fomdgt3_1),200). orientation(fomdgt3_6,130,200). coords(fomdgt3_6,210,80,200).  visible(fomdgt3_6,200). happensAt(walking(fomdgt3_6),200).
orientation(fomdgt3_1,150,201). coords(fomdgt3_1,77,98,201).  visible(fomdgt3_1,201). happensAt(walking(fomdgt3_1),201). orientation(fomdgt3_6,130,201). coords(fomdgt3_6,208,81,201).  visible(fomdgt3_6,201). happensAt(walking(fomdgt3_6),201).
orientation(fomdgt3_1,150,202). coords(fomdgt3_1,77,98,202).  visible(fomdgt3_1,202). happensAt(walking(fomdgt3_1),202). orientation(fomdgt3_6,130,202). coords(fomdgt3_6,208,81,202).  visible(fomdgt3_6,202). happensAt(walking(fomdgt3_6),202).
orientation(fomdgt3_1,150,203). coords(fomdgt3_1,77,98,203).  visible(fomdgt3_1,203). happensAt(walking(fomdgt3_1),203). orientation(fomdgt3_6,130,203). coords(fomdgt3_6,208,81,203).  visible(fomdgt3_6,203). happensAt(walking(fomdgt3_6),203).
orientation(fomdgt3_1,150,204). coords(fomdgt3_1,77,98,204).  visible(fomdgt3_1,204). happensAt(walking(fomdgt3_1),204). orientation(fomdgt3_6,130,204). coords(fomdgt3_6,208,81,204).  visible(fomdgt3_6,204). happensAt(walking(fomdgt3_6),204).
orientation(fomdgt3_1,150,205). coords(fomdgt3_1,77,98,205).  visible(fomdgt3_1,205). happensAt(walking(fomdgt3_1),205). orientation(fomdgt3_6,130,205). coords(fomdgt3_6,208,81,205).  visible(fomdgt3_6,205). happensAt(walking(fomdgt3_6),205).
orientation(fomdgt3_1,150,206). coords(fomdgt3_1,77,98,206).  visible(fomdgt3_1,206). happensAt(walking(fomdgt3_1),206). orientation(fomdgt3_6,130,206). coords(fomdgt3_6,208,81,206).  visible(fomdgt3_6,206). happensAt(walking(fomdgt3_6),206).
orientation(fomdgt3_1,150,207). coords(fomdgt3_1,77,98,207).  visible(fomdgt3_1,207). happensAt(walking(fomdgt3_1),207). orientation(fomdgt3_6,130,207). coords(fomdgt3_6,208,81,207).  visible(fomdgt3_6,207). happensAt(walking(fomdgt3_6),207).
orientation(fomdgt3_1,150,208). coords(fomdgt3_1,74,98,208).  visible(fomdgt3_1,208). happensAt(walking(fomdgt3_1),208). orientation(fomdgt3_6,130,208). coords(fomdgt3_6,208,84,208).  visible(fomdgt3_6,208). happensAt(walking(fomdgt3_6),208).
orientation(fomdgt3_1,150,209). coords(fomdgt3_1,74,98,209).  visible(fomdgt3_1,209). happensAt(walking(fomdgt3_1),209). orientation(fomdgt3_6,130,209). coords(fomdgt3_6,208,84,209).  visible(fomdgt3_6,209). happensAt(walking(fomdgt3_6),209).
orientation(fomdgt3_1,150,210). coords(fomdgt3_1,74,98,210).  visible(fomdgt3_1,210). happensAt(walking(fomdgt3_1),210). orientation(fomdgt3_6,130,210). coords(fomdgt3_6,208,85,210).  visible(fomdgt3_6,210). happensAt(walking(fomdgt3_6),210).
#show holdsAt/2.