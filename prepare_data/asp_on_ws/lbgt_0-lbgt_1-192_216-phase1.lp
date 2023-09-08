
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

	orientation(lbgt_0,156,192). coords(lbgt_0,206,182,192).  visible(lbgt_0,192). happensAt(walking(lbgt_0),192). orientation(lbgt_1,152,192). coords(lbgt_1,79,95,192).  visible(lbgt_1,192). happensAt(walking(lbgt_1),192).
orientation(lbgt_0,153,193). coords(lbgt_0,204,182,193).  visible(lbgt_0,193). happensAt(walking(lbgt_0),193). orientation(lbgt_1,152,193). coords(lbgt_1,79,95,193).  visible(lbgt_1,193). happensAt(walking(lbgt_1),193).
orientation(lbgt_0,150,194). coords(lbgt_0,204,181,194).  visible(lbgt_0,194). happensAt(walking(lbgt_0),194). orientation(lbgt_1,152,194). coords(lbgt_1,80,94,194).  visible(lbgt_1,194). happensAt(walking(lbgt_1),194).
orientation(lbgt_0,150,195). coords(lbgt_0,203,180,195).  visible(lbgt_0,195). happensAt(walking(lbgt_0),195). orientation(lbgt_1,147,195). coords(lbgt_1,80,94,195).  visible(lbgt_1,195). happensAt(walking(lbgt_1),195).
orientation(lbgt_0,150,196). coords(lbgt_0,202,179,196).  visible(lbgt_0,196). happensAt(walking(lbgt_0),196). orientation(lbgt_1,147,196). coords(lbgt_1,80,92,196).  visible(lbgt_1,196). happensAt(walking(lbgt_1),196).
orientation(lbgt_0,150,197). coords(lbgt_0,200,178,197).  visible(lbgt_0,197). happensAt(walking(lbgt_0),197). orientation(lbgt_1,147,197). coords(lbgt_1,80,92,197).  visible(lbgt_1,197). happensAt(walking(lbgt_1),197).
orientation(lbgt_0,150,198). coords(lbgt_0,199,176,198).  visible(lbgt_0,198). happensAt(walking(lbgt_0),198). orientation(lbgt_1,147,198). coords(lbgt_1,80,92,198).  visible(lbgt_1,198). happensAt(walking(lbgt_1),198).
orientation(lbgt_0,150,199). coords(lbgt_0,198,174,199).  visible(lbgt_0,199). happensAt(walking(lbgt_0),199). orientation(lbgt_1,147,199). coords(lbgt_1,80,91,199).  visible(lbgt_1,199). happensAt(walking(lbgt_1),199).
orientation(lbgt_0,150,200). coords(lbgt_0,196,172,200).  visible(lbgt_0,200). happensAt(walking(lbgt_0),200). orientation(lbgt_1,147,200). coords(lbgt_1,80,90,200).  visible(lbgt_1,200). happensAt(walking(lbgt_1),200).
orientation(lbgt_0,150,201). coords(lbgt_0,194,169,201).  visible(lbgt_0,201). happensAt(walking(lbgt_0),201). orientation(lbgt_1,147,201). coords(lbgt_1,80,89,201).  visible(lbgt_1,201). happensAt(walking(lbgt_1),201).
orientation(lbgt_0,150,202). coords(lbgt_0,192,167,202).  visible(lbgt_0,202). happensAt(walking(lbgt_0),202). orientation(lbgt_1,147,202). coords(lbgt_1,80,89,202).  visible(lbgt_1,202). happensAt(walking(lbgt_1),202).
orientation(lbgt_0,150,203). coords(lbgt_0,190,166,203).  visible(lbgt_0,203). happensAt(walking(lbgt_0),203). orientation(lbgt_1,147,203). coords(lbgt_1,80,89,203).  visible(lbgt_1,203). happensAt(walking(lbgt_1),203).
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
#show holdsAt/2.