
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

	orientation(fra1gt_6,101,186). coords(fra1gt_6,285,4,186).  visible(fra1gt_6,186). happensAt(walking(fra1gt_6),186). orientation(fra1gt_7,124,186). coords(fra1gt_7,169,21,186).  visible(fra1gt_7,186). happensAt(walking(fra1gt_7),186).
orientation(fra1gt_6,101,187). coords(fra1gt_6,284,5,187).  visible(fra1gt_6,187). happensAt(walking(fra1gt_6),187). orientation(fra1gt_7,125,187). coords(fra1gt_7,169,21,187).  visible(fra1gt_7,187). happensAt(walking(fra1gt_7),187).
orientation(fra1gt_6,101,188). coords(fra1gt_6,284,5,188).  visible(fra1gt_6,188). happensAt(walking(fra1gt_6),188). orientation(fra1gt_7,124,188). coords(fra1gt_7,169,21,188).  visible(fra1gt_7,188). happensAt(walking(fra1gt_7),188).
orientation(fra1gt_6,101,189). coords(fra1gt_6,284,5,189).  visible(fra1gt_6,189). happensAt(walking(fra1gt_6),189). orientation(fra1gt_7,125,189). coords(fra1gt_7,169,22,189).  visible(fra1gt_7,189). happensAt(walking(fra1gt_7),189).
orientation(fra1gt_6,101,190). coords(fra1gt_6,283,5,190).  visible(fra1gt_6,190). happensAt(walking(fra1gt_6),190). orientation(fra1gt_7,126,190). coords(fra1gt_7,169,21,190).  visible(fra1gt_7,190). happensAt(walking(fra1gt_7),190).
orientation(fra1gt_6,101,191). coords(fra1gt_6,282,5,191).  visible(fra1gt_6,191). happensAt(walking(fra1gt_6),191). orientation(fra1gt_7,127,191). coords(fra1gt_7,169,22,191).  visible(fra1gt_7,191). happensAt(walking(fra1gt_7),191).
orientation(fra1gt_6,101,192). coords(fra1gt_6,282,5,192).  visible(fra1gt_6,192). happensAt(walking(fra1gt_6),192). orientation(fra1gt_7,127,192). coords(fra1gt_7,168,22,192).  visible(fra1gt_7,192). happensAt(walking(fra1gt_7),192).
orientation(fra1gt_6,101,193). coords(fra1gt_6,282,5,193).  visible(fra1gt_6,193). happensAt(walking(fra1gt_6),193). orientation(fra1gt_7,127,193). coords(fra1gt_7,168,22,193).  visible(fra1gt_7,193). happensAt(walking(fra1gt_7),193).
orientation(fra1gt_6,101,194). coords(fra1gt_6,281,6,194).  visible(fra1gt_6,194). happensAt(walking(fra1gt_6),194). orientation(fra1gt_7,127,194). coords(fra1gt_7,168,22,194).  visible(fra1gt_7,194). happensAt(walking(fra1gt_7),194).
orientation(fra1gt_6,101,195). coords(fra1gt_6,281,7,195).  visible(fra1gt_6,195). happensAt(walking(fra1gt_6),195). orientation(fra1gt_7,127,195). coords(fra1gt_7,168,23,195).  visible(fra1gt_7,195). happensAt(walking(fra1gt_7),195).
orientation(fra1gt_6,101,196). coords(fra1gt_6,280,7,196).  visible(fra1gt_6,196). happensAt(walking(fra1gt_6),196). orientation(fra1gt_7,127,196). coords(fra1gt_7,168,23,196).  visible(fra1gt_7,196). happensAt(walking(fra1gt_7),196).
orientation(fra1gt_6,101,197). coords(fra1gt_6,280,8,197).  visible(fra1gt_6,197). happensAt(walking(fra1gt_6),197). orientation(fra1gt_7,127,197). coords(fra1gt_7,168,23,197).  visible(fra1gt_7,197). happensAt(walking(fra1gt_7),197).
orientation(fra1gt_6,101,198). coords(fra1gt_6,280,8,198).  visible(fra1gt_6,198). happensAt(walking(fra1gt_6),198). orientation(fra1gt_7,127,198). coords(fra1gt_7,168,23,198).  visible(fra1gt_7,198). happensAt(walking(fra1gt_7),198).
orientation(fra1gt_6,101,199). coords(fra1gt_6,280,8,199).  visible(fra1gt_6,199). happensAt(walking(fra1gt_6),199). orientation(fra1gt_7,125,199). coords(fra1gt_7,167,24,199).  visible(fra1gt_7,199). happensAt(walking(fra1gt_7),199).
orientation(fra1gt_6,101,200). coords(fra1gt_6,280,8,200).  visible(fra1gt_6,200). happensAt(walking(fra1gt_6),200). orientation(fra1gt_7,127,200). coords(fra1gt_7,167,26,200).  visible(fra1gt_7,200). happensAt(walking(fra1gt_7),200).
orientation(fra1gt_6,101,201). coords(fra1gt_6,279,8,201).  visible(fra1gt_6,201). happensAt(walking(fra1gt_6),201). orientation(fra1gt_7,127,201). coords(fra1gt_7,166,26,201).  visible(fra1gt_7,201). happensAt(walking(fra1gt_7),201).
orientation(fra1gt_6,101,202). coords(fra1gt_6,279,8,202).  visible(fra1gt_6,202). happensAt(walking(fra1gt_6),202). orientation(fra1gt_7,127,202). coords(fra1gt_7,166,26,202).  visible(fra1gt_7,202). happensAt(walking(fra1gt_7),202).
orientation(fra1gt_6,101,203). coords(fra1gt_6,278,8,203).  visible(fra1gt_6,203). happensAt(walking(fra1gt_6),203). orientation(fra1gt_7,127,203). coords(fra1gt_7,166,26,203).  visible(fra1gt_7,203). happensAt(walking(fra1gt_7),203).
orientation(fra1gt_6,101,204). coords(fra1gt_6,278,8,204).  visible(fra1gt_6,204). happensAt(walking(fra1gt_6),204). orientation(fra1gt_7,127,204). coords(fra1gt_7,166,27,204).  visible(fra1gt_7,204). happensAt(walking(fra1gt_7),204).
orientation(fra1gt_6,101,205). coords(fra1gt_6,278,10,205).  visible(fra1gt_6,205). happensAt(walking(fra1gt_6),205). orientation(fra1gt_7,127,205). coords(fra1gt_7,166,27,205).  visible(fra1gt_7,205). happensAt(walking(fra1gt_7),205).
orientation(fra1gt_6,101,206). coords(fra1gt_6,277,11,206).  visible(fra1gt_6,206). happensAt(walking(fra1gt_6),206). orientation(fra1gt_7,127,206). coords(fra1gt_7,165,28,206).  visible(fra1gt_7,206). happensAt(walking(fra1gt_7),206).
orientation(fra1gt_6,101,207). coords(fra1gt_6,277,12,207).  visible(fra1gt_6,207). happensAt(walking(fra1gt_6),207). orientation(fra1gt_7,127,207). coords(fra1gt_7,164,28,207).  visible(fra1gt_7,207). happensAt(walking(fra1gt_7),207).
orientation(fra1gt_6,101,208). coords(fra1gt_6,276,12,208).  visible(fra1gt_6,208). happensAt(walking(fra1gt_6),208). orientation(fra1gt_7,127,208). coords(fra1gt_7,163,28,208).  visible(fra1gt_7,208). happensAt(walking(fra1gt_7),208).
orientation(fra1gt_6,101,209). coords(fra1gt_6,276,13,209).  visible(fra1gt_6,209). happensAt(walking(fra1gt_6),209). orientation(fra1gt_7,127,209). coords(fra1gt_7,163,28,209).  visible(fra1gt_7,209). happensAt(walking(fra1gt_7),209).
orientation(fra1gt_6,101,210). coords(fra1gt_6,276,13,210).  visible(fra1gt_6,210). happensAt(walking(fra1gt_6),210). orientation(fra1gt_7,127,210). coords(fra1gt_7,163,28,210).  visible(fra1gt_7,210). happensAt(walking(fra1gt_7),210).
#show holdsAt/2.