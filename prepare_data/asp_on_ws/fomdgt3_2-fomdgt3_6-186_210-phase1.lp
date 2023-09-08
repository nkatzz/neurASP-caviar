
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

	orientation(fomdgt3_2,130,186). coords(fomdgt3_2,239,168,186).  visible(fomdgt3_2,186). happensAt(walking(fomdgt3_2),186). orientation(fomdgt3_6,130,186). coords(fomdgt3_6,220,63,186).  visible(fomdgt3_6,186). happensAt(walking(fomdgt3_6),186).
orientation(fomdgt3_2,130,187). coords(fomdgt3_2,239,165,187).  visible(fomdgt3_2,187). happensAt(walking(fomdgt3_2),187). orientation(fomdgt3_6,130,187). coords(fomdgt3_6,220,64,187).  visible(fomdgt3_6,187). happensAt(walking(fomdgt3_6),187).
orientation(fomdgt3_2,130,188). coords(fomdgt3_2,239,164,188).  visible(fomdgt3_2,188). happensAt(walking(fomdgt3_2),188). orientation(fomdgt3_6,130,188). coords(fomdgt3_6,220,66,188).  visible(fomdgt3_6,188). happensAt(walking(fomdgt3_6),188).
orientation(fomdgt3_2,130,189). coords(fomdgt3_2,239,163,189).  visible(fomdgt3_2,189). happensAt(walking(fomdgt3_2),189). orientation(fomdgt3_6,130,189). coords(fomdgt3_6,220,68,189).  visible(fomdgt3_6,189). happensAt(walking(fomdgt3_6),189).
orientation(fomdgt3_2,130,190). coords(fomdgt3_2,237,158,190).  visible(fomdgt3_2,190). happensAt(walking(fomdgt3_2),190). orientation(fomdgt3_6,130,190). coords(fomdgt3_6,219,69,190).  visible(fomdgt3_6,190). happensAt(walking(fomdgt3_6),190).
orientation(fomdgt3_2,130,191). coords(fomdgt3_2,237,156,191).  visible(fomdgt3_2,191). happensAt(walking(fomdgt3_2),191). orientation(fomdgt3_6,130,191). coords(fomdgt3_6,218,70,191).  visible(fomdgt3_6,191). happensAt(walking(fomdgt3_6),191).
orientation(fomdgt3_2,130,192). coords(fomdgt3_2,237,153,192).  visible(fomdgt3_2,192). happensAt(walking(fomdgt3_2),192). orientation(fomdgt3_6,130,192). coords(fomdgt3_6,218,71,192).  visible(fomdgt3_6,192). happensAt(walking(fomdgt3_6),192).
orientation(fomdgt3_2,130,193). coords(fomdgt3_2,236,153,193).  visible(fomdgt3_2,193). happensAt(walking(fomdgt3_2),193). orientation(fomdgt3_6,130,193). coords(fomdgt3_6,218,70,193).  visible(fomdgt3_6,193). happensAt(walking(fomdgt3_6),193).
orientation(fomdgt3_2,130,194). coords(fomdgt3_2,235,150,194).  visible(fomdgt3_2,194). happensAt(walking(fomdgt3_2),194). orientation(fomdgt3_6,130,194). coords(fomdgt3_6,217,72,194).  visible(fomdgt3_6,194). happensAt(walking(fomdgt3_6),194).
orientation(fomdgt3_2,130,195). coords(fomdgt3_2,235,149,195).  visible(fomdgt3_2,195). happensAt(walking(fomdgt3_2),195). orientation(fomdgt3_6,130,195). coords(fomdgt3_6,216,72,195).  visible(fomdgt3_6,195). happensAt(walking(fomdgt3_6),195).
orientation(fomdgt3_2,130,196). coords(fomdgt3_2,234,148,196).  visible(fomdgt3_2,196). happensAt(walking(fomdgt3_2),196). orientation(fomdgt3_6,130,196). coords(fomdgt3_6,215,72,196).  visible(fomdgt3_6,196). happensAt(walking(fomdgt3_6),196).
orientation(fomdgt3_2,130,197). coords(fomdgt3_2,234,148,197).  visible(fomdgt3_2,197). happensAt(walking(fomdgt3_2),197). orientation(fomdgt3_6,130,197). coords(fomdgt3_6,212,75,197).  visible(fomdgt3_6,197). happensAt(walking(fomdgt3_6),197).
orientation(fomdgt3_2,130,198). coords(fomdgt3_2,234,148,198).  visible(fomdgt3_2,198). happensAt(walking(fomdgt3_2),198). orientation(fomdgt3_6,130,198). coords(fomdgt3_6,211,79,198).  visible(fomdgt3_6,198). happensAt(walking(fomdgt3_6),198).
orientation(fomdgt3_2,130,199). coords(fomdgt3_2,234,148,199).  visible(fomdgt3_2,199). happensAt(walking(fomdgt3_2),199). orientation(fomdgt3_6,130,199). coords(fomdgt3_6,211,79,199).  visible(fomdgt3_6,199). happensAt(walking(fomdgt3_6),199).
orientation(fomdgt3_2,130,200). coords(fomdgt3_2,233,145,200).  visible(fomdgt3_2,200). happensAt(walking(fomdgt3_2),200). orientation(fomdgt3_6,130,200). coords(fomdgt3_6,210,80,200).  visible(fomdgt3_6,200). happensAt(walking(fomdgt3_6),200).
orientation(fomdgt3_2,130,201). coords(fomdgt3_2,232,144,201).  visible(fomdgt3_2,201). happensAt(walking(fomdgt3_2),201). orientation(fomdgt3_6,130,201). coords(fomdgt3_6,208,81,201).  visible(fomdgt3_6,201). happensAt(walking(fomdgt3_6),201).
orientation(fomdgt3_2,130,202). coords(fomdgt3_2,232,143,202).  visible(fomdgt3_2,202). happensAt(walking(fomdgt3_2),202). orientation(fomdgt3_6,130,202). coords(fomdgt3_6,208,81,202).  visible(fomdgt3_6,202). happensAt(walking(fomdgt3_6),202).
orientation(fomdgt3_2,130,203). coords(fomdgt3_2,232,142,203).  visible(fomdgt3_2,203). happensAt(walking(fomdgt3_2),203). orientation(fomdgt3_6,130,203). coords(fomdgt3_6,208,81,203).  visible(fomdgt3_6,203). happensAt(walking(fomdgt3_6),203).
orientation(fomdgt3_2,130,204). coords(fomdgt3_2,232,142,204).  visible(fomdgt3_2,204). happensAt(walking(fomdgt3_2),204). orientation(fomdgt3_6,130,204). coords(fomdgt3_6,208,81,204).  visible(fomdgt3_6,204). happensAt(walking(fomdgt3_6),204).
orientation(fomdgt3_2,130,205). coords(fomdgt3_2,231,141,205).  visible(fomdgt3_2,205). happensAt(walking(fomdgt3_2),205). orientation(fomdgt3_6,130,205). coords(fomdgt3_6,208,81,205).  visible(fomdgt3_6,205). happensAt(walking(fomdgt3_6),205).
orientation(fomdgt3_2,130,206). coords(fomdgt3_2,231,134,206).  visible(fomdgt3_2,206). happensAt(walking(fomdgt3_2),206). orientation(fomdgt3_6,130,206). coords(fomdgt3_6,208,81,206).  visible(fomdgt3_6,206). happensAt(walking(fomdgt3_6),206).
orientation(fomdgt3_2,130,207). coords(fomdgt3_2,231,134,207).  visible(fomdgt3_2,207). happensAt(walking(fomdgt3_2),207). orientation(fomdgt3_6,130,207). coords(fomdgt3_6,208,81,207).  visible(fomdgt3_6,207). happensAt(walking(fomdgt3_6),207).
orientation(fomdgt3_2,130,208). coords(fomdgt3_2,231,134,208).  visible(fomdgt3_2,208). happensAt(walking(fomdgt3_2),208). orientation(fomdgt3_6,130,208). coords(fomdgt3_6,208,84,208).  visible(fomdgt3_6,208). happensAt(walking(fomdgt3_6),208).
orientation(fomdgt3_2,130,209). coords(fomdgt3_2,231,134,209).  visible(fomdgt3_2,209). happensAt(walking(fomdgt3_2),209). orientation(fomdgt3_6,130,209). coords(fomdgt3_6,208,84,209).  visible(fomdgt3_6,209). happensAt(walking(fomdgt3_6),209).
orientation(fomdgt3_2,130,210). coords(fomdgt3_2,231,130,210).  visible(fomdgt3_2,210). happensAt(walking(fomdgt3_2),210). orientation(fomdgt3_6,130,210). coords(fomdgt3_6,208,85,210).  visible(fomdgt3_6,210). happensAt(walking(fomdgt3_6),210).
#show holdsAt/2.