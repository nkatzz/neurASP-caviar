
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

	orientation(mc1gt_0,147,192). coords(mc1gt_0,130,116,192).  visible(mc1gt_0,192). happensAt(walking(mc1gt_0),192). orientation(mc1gt_3,146,192). coords(mc1gt_3,118,82,192).  visible(mc1gt_3,192). happensAt(walking(mc1gt_3),192).
orientation(mc1gt_0,147,193). coords(mc1gt_0,130,117,193).  visible(mc1gt_0,193). happensAt(walking(mc1gt_0),193). orientation(mc1gt_3,146,193). coords(mc1gt_3,118,82,193).  visible(mc1gt_3,193). happensAt(walking(mc1gt_3),193).
orientation(mc1gt_0,147,194). coords(mc1gt_0,130,117,194).  visible(mc1gt_0,194). happensAt(walking(mc1gt_0),194). orientation(mc1gt_3,146,194). coords(mc1gt_3,119,82,194).  visible(mc1gt_3,194). happensAt(walking(mc1gt_3),194).
orientation(mc1gt_0,147,195). coords(mc1gt_0,131,117,195).  visible(mc1gt_0,195). happensAt(walking(mc1gt_0),195). orientation(mc1gt_3,146,195). coords(mc1gt_3,120,82,195).  visible(mc1gt_3,195). happensAt(walking(mc1gt_3),195).
orientation(mc1gt_0,147,196). coords(mc1gt_0,131,117,196).  visible(mc1gt_0,196). happensAt(walking(mc1gt_0),196). orientation(mc1gt_3,146,196). coords(mc1gt_3,120,82,196).  visible(mc1gt_3,196). happensAt(walking(mc1gt_3),196).
orientation(mc1gt_0,147,197). coords(mc1gt_0,131,117,197).  visible(mc1gt_0,197). happensAt(walking(mc1gt_0),197). orientation(mc1gt_3,146,197). coords(mc1gt_3,121,82,197).  visible(mc1gt_3,197). happensAt(walking(mc1gt_3),197).
orientation(mc1gt_0,147,198). coords(mc1gt_0,131,117,198).  visible(mc1gt_0,198). happensAt(walking(mc1gt_0),198). orientation(mc1gt_3,146,198). coords(mc1gt_3,122,82,198).  visible(mc1gt_3,198). happensAt(walking(mc1gt_3),198).
orientation(mc1gt_0,147,199). coords(mc1gt_0,132,117,199).  visible(mc1gt_0,199). happensAt(walking(mc1gt_0),199). orientation(mc1gt_3,146,199). coords(mc1gt_3,123,82,199).  visible(mc1gt_3,199). happensAt(walking(mc1gt_3),199).
orientation(mc1gt_0,147,200). coords(mc1gt_0,134,117,200).  visible(mc1gt_0,200). happensAt(walking(mc1gt_0),200). orientation(mc1gt_3,146,200). coords(mc1gt_3,126,83,200).  visible(mc1gt_3,200). happensAt(walking(mc1gt_3),200).
orientation(mc1gt_0,147,201). coords(mc1gt_0,136,118,201).  visible(mc1gt_0,201). happensAt(walking(mc1gt_0),201). orientation(mc1gt_3,146,201). coords(mc1gt_3,127,83,201).  visible(mc1gt_3,201). happensAt(walking(mc1gt_3),201).
orientation(mc1gt_0,147,202). coords(mc1gt_0,138,118,202).  visible(mc1gt_0,202). happensAt(walking(mc1gt_0),202). orientation(mc1gt_3,146,202). coords(mc1gt_3,129,83,202).  visible(mc1gt_3,202). happensAt(walking(mc1gt_3),202).
orientation(mc1gt_0,147,203). coords(mc1gt_0,139,118,203).  visible(mc1gt_0,203). happensAt(walking(mc1gt_0),203). orientation(mc1gt_3,146,203). coords(mc1gt_3,131,83,203).  visible(mc1gt_3,203). happensAt(walking(mc1gt_3),203).
orientation(mc1gt_0,147,204). coords(mc1gt_0,140,118,204).  visible(mc1gt_0,204). happensAt(walking(mc1gt_0),204). orientation(mc1gt_3,146,204). coords(mc1gt_3,133,84,204).  visible(mc1gt_3,204). happensAt(walking(mc1gt_3),204).
orientation(mc1gt_0,147,205). coords(mc1gt_0,141,119,205).  visible(mc1gt_0,205). happensAt(walking(mc1gt_0),205). orientation(mc1gt_3,146,205). coords(mc1gt_3,134,84,205).  visible(mc1gt_3,205). happensAt(walking(mc1gt_3),205).
orientation(mc1gt_0,147,206). coords(mc1gt_0,141,119,206).  visible(mc1gt_0,206). happensAt(walking(mc1gt_0),206). orientation(mc1gt_3,146,206). coords(mc1gt_3,136,84,206).  visible(mc1gt_3,206). happensAt(walking(mc1gt_3),206).
orientation(mc1gt_0,147,207). coords(mc1gt_0,142,119,207).  visible(mc1gt_0,207). happensAt(walking(mc1gt_0),207). orientation(mc1gt_3,146,207). coords(mc1gt_3,137,82,207).  visible(mc1gt_3,207). happensAt(walking(mc1gt_3),207).
orientation(mc1gt_0,147,208). coords(mc1gt_0,142,119,208).  visible(mc1gt_0,208). happensAt(walking(mc1gt_0),208). orientation(mc1gt_3,146,208). coords(mc1gt_3,138,82,208).  visible(mc1gt_3,208). happensAt(walking(mc1gt_3),208).
orientation(mc1gt_0,147,209). coords(mc1gt_0,142,119,209).  visible(mc1gt_0,209). happensAt(walking(mc1gt_0),209). orientation(mc1gt_3,146,209). coords(mc1gt_3,138,83,209).  visible(mc1gt_3,209). happensAt(walking(mc1gt_3),209).
orientation(mc1gt_0,147,210). coords(mc1gt_0,143,119,210).  visible(mc1gt_0,210). happensAt(walking(mc1gt_0),210). orientation(mc1gt_3,146,210). coords(mc1gt_3,139,83,210).  visible(mc1gt_3,210). happensAt(walking(mc1gt_3),210).
orientation(mc1gt_0,147,211). coords(mc1gt_0,143,120,211).  visible(mc1gt_0,211). happensAt(walking(mc1gt_0),211). orientation(mc1gt_3,146,211). coords(mc1gt_3,141,83,211).  visible(mc1gt_3,211). happensAt(walking(mc1gt_3),211).
orientation(mc1gt_0,147,212). coords(mc1gt_0,143,121,212).  visible(mc1gt_0,212). happensAt(walking(mc1gt_0),212). orientation(mc1gt_3,146,212). coords(mc1gt_3,142,83,212).  visible(mc1gt_3,212). happensAt(walking(mc1gt_3),212).
orientation(mc1gt_0,147,213). coords(mc1gt_0,144,122,213).  visible(mc1gt_0,213). happensAt(walking(mc1gt_0),213). orientation(mc1gt_3,146,213). coords(mc1gt_3,142,83,213).  visible(mc1gt_3,213). happensAt(walking(mc1gt_3),213).
orientation(mc1gt_0,147,214). coords(mc1gt_0,144,123,214).  visible(mc1gt_0,214). happensAt(walking(mc1gt_0),214). orientation(mc1gt_3,146,214). coords(mc1gt_3,143,83,214).  visible(mc1gt_3,214). happensAt(walking(mc1gt_3),214).
orientation(mc1gt_0,147,215). coords(mc1gt_0,146,125,215).  visible(mc1gt_0,215). happensAt(walking(mc1gt_0),215). orientation(mc1gt_3,146,215). coords(mc1gt_3,144,84,215).  visible(mc1gt_3,215). happensAt(walking(mc1gt_3),215).
orientation(mc1gt_0,147,216). coords(mc1gt_0,147,126,216).  visible(mc1gt_0,216). happensAt(walking(mc1gt_0),216). orientation(mc1gt_3,146,216). coords(mc1gt_3,145,84,216).  visible(mc1gt_3,216). happensAt(walking(mc1gt_3),216).
#show holdsAt/2.