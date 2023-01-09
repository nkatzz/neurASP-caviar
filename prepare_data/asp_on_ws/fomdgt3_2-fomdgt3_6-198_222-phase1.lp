
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
orientation(fomdgt3_2,130,211). coords(fomdgt3_2,228,130,211).  visible(fomdgt3_2,211). happensAt(walking(fomdgt3_2),211). orientation(fomdgt3_6,130,211). coords(fomdgt3_6,208,86,211).  visible(fomdgt3_6,211). happensAt(walking(fomdgt3_6),211).
orientation(fomdgt3_2,130,212). coords(fomdgt3_2,228,130,212).  visible(fomdgt3_2,212). happensAt(walking(fomdgt3_2),212). orientation(fomdgt3_6,130,212). coords(fomdgt3_6,208,86,212).  visible(fomdgt3_6,212). happensAt(walking(fomdgt3_6),212).
orientation(fomdgt3_2,130,213). coords(fomdgt3_2,228,130,213).  visible(fomdgt3_2,213). happensAt(walking(fomdgt3_2),213). orientation(fomdgt3_6,130,213). coords(fomdgt3_6,209,87,213).  visible(fomdgt3_6,213). happensAt(walking(fomdgt3_6),213).
orientation(fomdgt3_2,130,214). coords(fomdgt3_2,228,130,214).  visible(fomdgt3_2,214). happensAt(walking(fomdgt3_2),214). orientation(fomdgt3_6,130,214). coords(fomdgt3_6,209,87,214).  visible(fomdgt3_6,214). happensAt(walking(fomdgt3_6),214).
orientation(fomdgt3_2,130,215). coords(fomdgt3_2,228,130,215).  visible(fomdgt3_2,215). happensAt(walking(fomdgt3_2),215). orientation(fomdgt3_6,130,215). coords(fomdgt3_6,210,91,215).  visible(fomdgt3_6,215). happensAt(walking(fomdgt3_6),215).
orientation(fomdgt3_2,130,216). coords(fomdgt3_2,228,128,216).  visible(fomdgt3_2,216). happensAt(walking(fomdgt3_2),216). orientation(fomdgt3_6,130,216). coords(fomdgt3_6,210,91,216).  visible(fomdgt3_6,216). happensAt(walking(fomdgt3_6),216).
orientation(fomdgt3_2,130,217). coords(fomdgt3_2,228,128,217).  visible(fomdgt3_2,217). happensAt(walking(fomdgt3_2),217). orientation(fomdgt3_6,130,217). coords(fomdgt3_6,210,91,217).  visible(fomdgt3_6,217). happensAt(walking(fomdgt3_6),217).
orientation(fomdgt3_2,130,218). coords(fomdgt3_2,228,128,218).  visible(fomdgt3_2,218). happensAt(walking(fomdgt3_2),218). orientation(fomdgt3_6,130,218). coords(fomdgt3_6,210,91,218).  visible(fomdgt3_6,218). happensAt(walking(fomdgt3_6),218).
orientation(fomdgt3_2,130,219). coords(fomdgt3_2,228,128,219).  visible(fomdgt3_2,219). happensAt(walking(fomdgt3_2),219). orientation(fomdgt3_6,130,219). coords(fomdgt3_6,210,91,219).  visible(fomdgt3_6,219). happensAt(walking(fomdgt3_6),219).
orientation(fomdgt3_2,130,220). coords(fomdgt3_2,228,128,220).  visible(fomdgt3_2,220). happensAt(walking(fomdgt3_2),220). orientation(fomdgt3_6,130,220). coords(fomdgt3_6,205,91,220).  visible(fomdgt3_6,220). happensAt(walking(fomdgt3_6),220).
orientation(fomdgt3_2,130,221). coords(fomdgt3_2,228,128,221).  visible(fomdgt3_2,221). happensAt(walking(fomdgt3_2),221). orientation(fomdgt3_6,130,221). coords(fomdgt3_6,205,89,221).  visible(fomdgt3_6,221). happensAt(walking(fomdgt3_6),221).
orientation(fomdgt3_2,130,222). coords(fomdgt3_2,228,128,222).  visible(fomdgt3_2,222). happensAt(walking(fomdgt3_2),222). orientation(fomdgt3_6,130,222). coords(fomdgt3_6,205,89,222).  visible(fomdgt3_6,222). happensAt(walking(fomdgt3_6),222).
#show holdsAt/2.