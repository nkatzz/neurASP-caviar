
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

	orientation(fcgt_2,166,186). coords(fcgt_2,38,200,186).  visible(fcgt_2,186). happensAt(walking(fcgt_2),186). orientation(fcgt_3,159,186). coords(fcgt_3,103,163,186).  visible(fcgt_3,186). happensAt(inactive(fcgt_3),186).
orientation(fcgt_2,166,187). coords(fcgt_2,39,199,187).  visible(fcgt_2,187). happensAt(walking(fcgt_2),187). orientation(fcgt_3,159,187). coords(fcgt_3,103,163,187).  visible(fcgt_3,187). happensAt(inactive(fcgt_3),187).
orientation(fcgt_2,166,188). coords(fcgt_2,39,199,188).  visible(fcgt_2,188). happensAt(walking(fcgt_2),188). orientation(fcgt_3,159,188). coords(fcgt_3,103,163,188).  visible(fcgt_3,188). happensAt(inactive(fcgt_3),188).
orientation(fcgt_2,166,189). coords(fcgt_2,40,198,189).  visible(fcgt_2,189). happensAt(walking(fcgt_2),189). orientation(fcgt_3,159,189). coords(fcgt_3,103,163,189).  visible(fcgt_3,189). happensAt(inactive(fcgt_3),189).
orientation(fcgt_2,166,190). coords(fcgt_2,39,198,190).  visible(fcgt_2,190). happensAt(walking(fcgt_2),190). orientation(fcgt_3,159,190). coords(fcgt_3,103,163,190).  visible(fcgt_3,190). happensAt(inactive(fcgt_3),190).
orientation(fcgt_2,166,191). coords(fcgt_2,39,197,191).  visible(fcgt_2,191). happensAt(walking(fcgt_2),191). orientation(fcgt_3,159,191). coords(fcgt_3,103,163,191).  visible(fcgt_3,191). happensAt(inactive(fcgt_3),191).
orientation(fcgt_2,166,192). coords(fcgt_2,39,197,192).  visible(fcgt_2,192). happensAt(walking(fcgt_2),192). orientation(fcgt_3,159,192). coords(fcgt_3,103,163,192).  visible(fcgt_3,192). happensAt(inactive(fcgt_3),192).
orientation(fcgt_2,166,193). coords(fcgt_2,39,197,193).  visible(fcgt_2,193). happensAt(walking(fcgt_2),193). orientation(fcgt_3,159,193). coords(fcgt_3,103,163,193).  visible(fcgt_3,193). happensAt(inactive(fcgt_3),193).
orientation(fcgt_2,166,194). coords(fcgt_2,39,197,194).  visible(fcgt_2,194). happensAt(walking(fcgt_2),194). orientation(fcgt_3,159,194). coords(fcgt_3,103,163,194).  visible(fcgt_3,194). happensAt(inactive(fcgt_3),194).
orientation(fcgt_2,166,195). coords(fcgt_2,39,196,195).  visible(fcgt_2,195). happensAt(walking(fcgt_2),195). orientation(fcgt_3,159,195). coords(fcgt_3,103,163,195).  visible(fcgt_3,195). happensAt(inactive(fcgt_3),195).
orientation(fcgt_2,166,196). coords(fcgt_2,39,195,196).  visible(fcgt_2,196). happensAt(walking(fcgt_2),196). orientation(fcgt_3,159,196). coords(fcgt_3,103,163,196).  visible(fcgt_3,196). happensAt(inactive(fcgt_3),196).
orientation(fcgt_2,166,197). coords(fcgt_2,39,194,197).  visible(fcgt_2,197). happensAt(walking(fcgt_2),197). orientation(fcgt_3,159,197). coords(fcgt_3,103,163,197).  visible(fcgt_3,197). happensAt(inactive(fcgt_3),197).
orientation(fcgt_2,166,198). coords(fcgt_2,40,193,198).  visible(fcgt_2,198). happensAt(walking(fcgt_2),198). orientation(fcgt_3,159,198). coords(fcgt_3,103,163,198).  visible(fcgt_3,198). happensAt(inactive(fcgt_3),198).
orientation(fcgt_2,166,199). coords(fcgt_2,40,193,199).  visible(fcgt_2,199). happensAt(walking(fcgt_2),199). orientation(fcgt_3,159,199). coords(fcgt_3,103,163,199).  visible(fcgt_3,199). happensAt(inactive(fcgt_3),199).
orientation(fcgt_2,166,200). coords(fcgt_2,40,193,200).  visible(fcgt_2,200). happensAt(walking(fcgt_2),200). orientation(fcgt_3,159,200). coords(fcgt_3,103,163,200).  visible(fcgt_3,200). happensAt(inactive(fcgt_3),200).
orientation(fcgt_2,166,201). coords(fcgt_2,40,192,201).  visible(fcgt_2,201). happensAt(walking(fcgt_2),201). orientation(fcgt_3,159,201). coords(fcgt_3,102,163,201).  visible(fcgt_3,201). happensAt(inactive(fcgt_3),201).
orientation(fcgt_2,166,202). coords(fcgt_2,41,192,202).  visible(fcgt_2,202). happensAt(walking(fcgt_2),202). orientation(fcgt_3,159,202). coords(fcgt_3,102,163,202).  visible(fcgt_3,202). happensAt(walking(fcgt_3),202).
orientation(fcgt_2,166,203). coords(fcgt_2,41,191,203).  visible(fcgt_2,203). happensAt(walking(fcgt_2),203). orientation(fcgt_3,159,203). coords(fcgt_3,101,163,203).  visible(fcgt_3,203). happensAt(walking(fcgt_3),203).
orientation(fcgt_2,166,204). coords(fcgt_2,41,191,204).  visible(fcgt_2,204). happensAt(walking(fcgt_2),204). orientation(fcgt_3,159,204). coords(fcgt_3,101,164,204).  visible(fcgt_3,204). happensAt(walking(fcgt_3),204).
orientation(fcgt_2,166,205). coords(fcgt_2,42,190,205).  visible(fcgt_2,205). happensAt(walking(fcgt_2),205). orientation(fcgt_3,159,205). coords(fcgt_3,101,164,205).  visible(fcgt_3,205). happensAt(walking(fcgt_3),205).
orientation(fcgt_2,166,206). coords(fcgt_2,42,190,206).  visible(fcgt_2,206). happensAt(walking(fcgt_2),206). orientation(fcgt_3,159,206). coords(fcgt_3,101,165,206).  visible(fcgt_3,206). happensAt(walking(fcgt_3),206).
orientation(fcgt_2,166,207). coords(fcgt_2,42,190,207).  visible(fcgt_2,207). happensAt(walking(fcgt_2),207). orientation(fcgt_3,159,207). coords(fcgt_3,101,165,207).  visible(fcgt_3,207). happensAt(walking(fcgt_3),207).
orientation(fcgt_2,166,208). coords(fcgt_2,42,188,208).  visible(fcgt_2,208). happensAt(walking(fcgt_2),208). orientation(fcgt_3,153,208). coords(fcgt_3,100,166,208).  visible(fcgt_3,208). happensAt(walking(fcgt_3),208).
orientation(fcgt_2,158,209). coords(fcgt_2,42,188,209).  visible(fcgt_2,209). happensAt(walking(fcgt_2),209). orientation(fcgt_3,153,209). coords(fcgt_3,100,166,209).  visible(fcgt_3,209). happensAt(walking(fcgt_3),209).
orientation(fcgt_2,157,210). coords(fcgt_2,43,187,210).  visible(fcgt_2,210). happensAt(walking(fcgt_2),210). orientation(fcgt_3,149,210). coords(fcgt_3,100,166,210).  visible(fcgt_3,210). happensAt(walking(fcgt_3),210).
#show holdsAt/2.