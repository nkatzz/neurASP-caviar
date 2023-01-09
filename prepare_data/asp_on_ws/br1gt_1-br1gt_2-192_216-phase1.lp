
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

	orientation(br1gt_1,164,192). coords(br1gt_1,35,156,192).  visible(br1gt_1,192). happensAt(walking(br1gt_1),192). orientation(br1gt_2,171,192). coords(br1gt_2,86,207,192).  visible(br1gt_2,192). happensAt(inactive(br1gt_2),192).
orientation(br1gt_1,164,193). coords(br1gt_1,35,156,193).  visible(br1gt_1,193). happensAt(walking(br1gt_1),193). orientation(br1gt_2,171,193). coords(br1gt_2,86,207,193).  visible(br1gt_2,193). happensAt(inactive(br1gt_2),193).
orientation(br1gt_1,164,194). coords(br1gt_1,35,156,194).  visible(br1gt_1,194). happensAt(walking(br1gt_1),194). orientation(br1gt_2,171,194). coords(br1gt_2,86,207,194).  visible(br1gt_2,194). happensAt(inactive(br1gt_2),194).
orientation(br1gt_1,164,195). coords(br1gt_1,34,156,195).  visible(br1gt_1,195). happensAt(walking(br1gt_1),195). orientation(br1gt_2,171,195). coords(br1gt_2,86,207,195).  visible(br1gt_2,195). happensAt(inactive(br1gt_2),195).
orientation(br1gt_1,164,196). coords(br1gt_1,34,156,196).  visible(br1gt_1,196). happensAt(walking(br1gt_1),196). orientation(br1gt_2,171,196). coords(br1gt_2,86,207,196).  visible(br1gt_2,196). happensAt(inactive(br1gt_2),196).
orientation(br1gt_1,164,197). coords(br1gt_1,33,155,197).  visible(br1gt_1,197). happensAt(walking(br1gt_1),197). orientation(br1gt_2,171,197). coords(br1gt_2,86,207,197).  visible(br1gt_2,197). happensAt(inactive(br1gt_2),197).
orientation(br1gt_1,164,198). coords(br1gt_1,33,155,198).  visible(br1gt_1,198). happensAt(walking(br1gt_1),198). orientation(br1gt_2,171,198). coords(br1gt_2,86,207,198).  visible(br1gt_2,198). happensAt(inactive(br1gt_2),198).
orientation(br1gt_1,164,199). coords(br1gt_1,32,155,199).  visible(br1gt_1,199). happensAt(walking(br1gt_1),199). orientation(br1gt_2,171,199). coords(br1gt_2,86,207,199).  visible(br1gt_2,199). happensAt(inactive(br1gt_2),199).
orientation(br1gt_1,164,200). coords(br1gt_1,33,155,200).  visible(br1gt_1,200). happensAt(walking(br1gt_1),200). orientation(br1gt_2,171,200). coords(br1gt_2,86,207,200).  visible(br1gt_2,200). happensAt(inactive(br1gt_2),200).
orientation(br1gt_1,164,201). coords(br1gt_1,32,157,201).  visible(br1gt_1,201). happensAt(walking(br1gt_1),201). orientation(br1gt_2,171,201). coords(br1gt_2,86,207,201).  visible(br1gt_2,201). happensAt(inactive(br1gt_2),201).
orientation(br1gt_1,164,202). coords(br1gt_1,32,157,202).  visible(br1gt_1,202). happensAt(walking(br1gt_1),202). orientation(br1gt_2,171,202). coords(br1gt_2,86,207,202).  visible(br1gt_2,202). happensAt(inactive(br1gt_2),202).
orientation(br1gt_1,164,203). coords(br1gt_1,32,156,203).  visible(br1gt_1,203). happensAt(walking(br1gt_1),203). orientation(br1gt_2,171,203). coords(br1gt_2,86,207,203).  visible(br1gt_2,203). happensAt(inactive(br1gt_2),203).
orientation(br1gt_1,164,204). coords(br1gt_1,32,157,204).  visible(br1gt_1,204). happensAt(walking(br1gt_1),204). orientation(br1gt_2,171,204). coords(br1gt_2,86,207,204).  visible(br1gt_2,204). happensAt(inactive(br1gt_2),204).
orientation(br1gt_1,164,205). coords(br1gt_1,31,157,205).  visible(br1gt_1,205). happensAt(walking(br1gt_1),205). orientation(br1gt_2,171,205). coords(br1gt_2,86,207,205).  visible(br1gt_2,205). happensAt(inactive(br1gt_2),205).
orientation(br1gt_1,164,206). coords(br1gt_1,31,157,206).  visible(br1gt_1,206). happensAt(walking(br1gt_1),206). orientation(br1gt_2,171,206). coords(br1gt_2,86,207,206).  visible(br1gt_2,206). happensAt(inactive(br1gt_2),206).
orientation(br1gt_1,164,207). coords(br1gt_1,31,157,207).  visible(br1gt_1,207). happensAt(walking(br1gt_1),207). orientation(br1gt_2,171,207). coords(br1gt_2,86,207,207).  visible(br1gt_2,207). happensAt(inactive(br1gt_2),207).
orientation(br1gt_1,164,208). coords(br1gt_1,31,157,208).  visible(br1gt_1,208). happensAt(walking(br1gt_1),208). orientation(br1gt_2,171,208). coords(br1gt_2,86,207,208).  visible(br1gt_2,208). happensAt(inactive(br1gt_2),208).
orientation(br1gt_1,162,209). coords(br1gt_1,31,158,209).  visible(br1gt_1,209). happensAt(walking(br1gt_1),209). orientation(br1gt_2,171,209). coords(br1gt_2,86,207,209).  visible(br1gt_2,209). happensAt(inactive(br1gt_2),209).
orientation(br1gt_1,162,210). coords(br1gt_1,31,158,210).  visible(br1gt_1,210). happensAt(walking(br1gt_1),210). orientation(br1gt_2,171,210). coords(br1gt_2,86,207,210).  visible(br1gt_2,210). happensAt(inactive(br1gt_2),210).
orientation(br1gt_1,170,211). coords(br1gt_1,31,158,211).  visible(br1gt_1,211). happensAt(walking(br1gt_1),211). orientation(br1gt_2,171,211). coords(br1gt_2,86,207,211).  visible(br1gt_2,211). happensAt(inactive(br1gt_2),211).
orientation(br1gt_1,170,212). coords(br1gt_1,31,158,212).  visible(br1gt_1,212). happensAt(walking(br1gt_1),212). orientation(br1gt_2,171,212). coords(br1gt_2,86,207,212).  visible(br1gt_2,212). happensAt(inactive(br1gt_2),212).
orientation(br1gt_1,170,213). coords(br1gt_1,31,158,213).  visible(br1gt_1,213). happensAt(walking(br1gt_1),213). orientation(br1gt_2,171,213). coords(br1gt_2,86,207,213).  visible(br1gt_2,213). happensAt(inactive(br1gt_2),213).
orientation(br1gt_1,170,214). coords(br1gt_1,30,159,214).  visible(br1gt_1,214). happensAt(walking(br1gt_1),214). orientation(br1gt_2,171,214). coords(br1gt_2,86,207,214).  visible(br1gt_2,214). happensAt(inactive(br1gt_2),214).
orientation(br1gt_1,170,215). coords(br1gt_1,30,159,215).  visible(br1gt_1,215). happensAt(walking(br1gt_1),215). orientation(br1gt_2,171,215). coords(br1gt_2,86,207,215).  visible(br1gt_2,215). happensAt(inactive(br1gt_2),215).
orientation(br1gt_1,170,216). coords(br1gt_1,30,159,216).  visible(br1gt_1,216). happensAt(walking(br1gt_1),216). orientation(br1gt_2,171,216). coords(br1gt_2,86,207,216).  visible(br1gt_2,216). happensAt(inactive(br1gt_2),216).
#show holdsAt/2.