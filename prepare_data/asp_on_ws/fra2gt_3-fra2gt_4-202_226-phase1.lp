
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

	orientation(fra2gt_3,161,202). coords(fra2gt_3,43,177,202).  visible(fra2gt_3,202). happensAt(walking(fra2gt_3),202). orientation(fra2gt_4,37,202). coords(fra2gt_4,231,272,202).  visible(fra2gt_4,202). happensAt(walking(fra2gt_4),202).
orientation(fra2gt_3,161,203). coords(fra2gt_3,44,176,203).  visible(fra2gt_3,203). happensAt(walking(fra2gt_3),203). orientation(fra2gt_4,18,203). coords(fra2gt_4,230,270,203).  visible(fra2gt_4,203). happensAt(walking(fra2gt_4),203).
orientation(fra2gt_3,161,204). coords(fra2gt_3,44,175,204).  visible(fra2gt_3,204). happensAt(walking(fra2gt_3),204). orientation(fra2gt_4,18,204). coords(fra2gt_4,229,265,204).  visible(fra2gt_4,204). happensAt(walking(fra2gt_4),204).
orientation(fra2gt_3,161,205). coords(fra2gt_3,44,174,205).  visible(fra2gt_3,205). happensAt(walking(fra2gt_3),205). orientation(fra2gt_4,18,205). coords(fra2gt_4,229,261,205).  visible(fra2gt_4,205). happensAt(walking(fra2gt_4),205).
orientation(fra2gt_3,161,206). coords(fra2gt_3,44,174,206).  visible(fra2gt_3,206). happensAt(walking(fra2gt_3),206). orientation(fra2gt_4,18,206). coords(fra2gt_4,228,257,206).  visible(fra2gt_4,206). happensAt(walking(fra2gt_4),206).
orientation(fra2gt_3,161,207). coords(fra2gt_3,45,174,207).  visible(fra2gt_3,207). happensAt(walking(fra2gt_3),207). orientation(fra2gt_4,18,207). coords(fra2gt_4,226,256,207).  visible(fra2gt_4,207). happensAt(walking(fra2gt_4),207).
orientation(fra2gt_3,161,208). coords(fra2gt_3,45,173,208).  visible(fra2gt_3,208). happensAt(walking(fra2gt_3),208). orientation(fra2gt_4,18,208). coords(fra2gt_4,225,255,208).  visible(fra2gt_4,208). happensAt(walking(fra2gt_4),208).
orientation(fra2gt_3,161,209). coords(fra2gt_3,45,173,209).  visible(fra2gt_3,209). happensAt(walking(fra2gt_3),209). orientation(fra2gt_4,18,209). coords(fra2gt_4,224,253,209).  visible(fra2gt_4,209). happensAt(walking(fra2gt_4),209).
orientation(fra2gt_3,161,210). coords(fra2gt_3,46,172,210).  visible(fra2gt_3,210). happensAt(walking(fra2gt_3),210). orientation(fra2gt_4,18,210). coords(fra2gt_4,222,251,210).  visible(fra2gt_4,210). happensAt(walking(fra2gt_4),210).
orientation(fra2gt_3,161,211). coords(fra2gt_3,46,171,211).  visible(fra2gt_3,211). happensAt(walking(fra2gt_3),211). orientation(fra2gt_4,18,211). coords(fra2gt_4,220,248,211).  visible(fra2gt_4,211). happensAt(walking(fra2gt_4),211).
orientation(fra2gt_3,161,212). coords(fra2gt_3,46,170,212).  visible(fra2gt_3,212). happensAt(walking(fra2gt_3),212). orientation(fra2gt_4,7,212). coords(fra2gt_4,218,245,212).  visible(fra2gt_4,212). happensAt(walking(fra2gt_4),212).
orientation(fra2gt_3,161,213). coords(fra2gt_3,46,169,213).  visible(fra2gt_3,213). happensAt(walking(fra2gt_3),213). orientation(fra2gt_4,7,213). coords(fra2gt_4,215,243,213).  visible(fra2gt_4,213). happensAt(walking(fra2gt_4),213).
orientation(fra2gt_3,161,214). coords(fra2gt_3,46,168,214).  visible(fra2gt_3,214). happensAt(walking(fra2gt_3),214). orientation(fra2gt_4,7,214). coords(fra2gt_4,212,242,214).  visible(fra2gt_4,214). happensAt(walking(fra2gt_4),214).
orientation(fra2gt_3,161,215). coords(fra2gt_3,46,167,215).  visible(fra2gt_3,215). happensAt(walking(fra2gt_3),215). orientation(fra2gt_4,7,215). coords(fra2gt_4,211,239,215).  visible(fra2gt_4,215). happensAt(walking(fra2gt_4),215).
orientation(fra2gt_3,161,216). coords(fra2gt_3,47,166,216).  visible(fra2gt_3,216). happensAt(walking(fra2gt_3),216). orientation(fra2gt_4,7,216). coords(fra2gt_4,210,235,216).  visible(fra2gt_4,216). happensAt(walking(fra2gt_4),216).
orientation(fra2gt_3,161,217). coords(fra2gt_3,48,166,217).  visible(fra2gt_3,217). happensAt(walking(fra2gt_3),217). orientation(fra2gt_4,7,217). coords(fra2gt_4,209,232,217).  visible(fra2gt_4,217). happensAt(walking(fra2gt_4),217).
orientation(fra2gt_3,161,218). coords(fra2gt_3,49,166,218).  visible(fra2gt_3,218). happensAt(walking(fra2gt_3),218). orientation(fra2gt_4,170,218). coords(fra2gt_4,208,229,218).  visible(fra2gt_4,218). happensAt(walking(fra2gt_4),218).
orientation(fra2gt_3,161,219). coords(fra2gt_3,49,166,219).  visible(fra2gt_3,219). happensAt(walking(fra2gt_3),219). orientation(fra2gt_4,170,219). coords(fra2gt_4,207,228,219).  visible(fra2gt_4,219). happensAt(walking(fra2gt_4),219).
orientation(fra2gt_3,161,220). coords(fra2gt_3,49,165,220).  visible(fra2gt_3,220). happensAt(walking(fra2gt_3),220). orientation(fra2gt_4,170,220). coords(fra2gt_4,207,228,220).  visible(fra2gt_4,220). happensAt(walking(fra2gt_4),220).
orientation(fra2gt_3,161,221). coords(fra2gt_3,50,165,221).  visible(fra2gt_3,221). happensAt(walking(fra2gt_3),221). orientation(fra2gt_4,170,221). coords(fra2gt_4,205,226,221).  visible(fra2gt_4,221). happensAt(walking(fra2gt_4),221).
orientation(fra2gt_3,161,222). coords(fra2gt_3,50,165,222).  visible(fra2gt_3,222). happensAt(walking(fra2gt_3),222). orientation(fra2gt_4,170,222). coords(fra2gt_4,203,225,222).  visible(fra2gt_4,222). happensAt(walking(fra2gt_4),222).
orientation(fra2gt_3,161,223). coords(fra2gt_3,50,164,223).  visible(fra2gt_3,223). happensAt(walking(fra2gt_3),223). orientation(fra2gt_4,170,223). coords(fra2gt_4,201,223,223).  visible(fra2gt_4,223). happensAt(walking(fra2gt_4),223).
orientation(fra2gt_3,161,224). coords(fra2gt_3,50,163,224).  visible(fra2gt_3,224). happensAt(walking(fra2gt_3),224). orientation(fra2gt_4,170,224). coords(fra2gt_4,198,218,224).  visible(fra2gt_4,224). happensAt(walking(fra2gt_4),224).
orientation(fra2gt_3,161,225). coords(fra2gt_3,50,162,225).  visible(fra2gt_3,225). happensAt(walking(fra2gt_3),225). orientation(fra2gt_4,163,225). coords(fra2gt_4,197,214,225).  visible(fra2gt_4,225). happensAt(walking(fra2gt_4),225).
orientation(fra2gt_3,161,226). coords(fra2gt_3,51,161,226).  visible(fra2gt_3,226). happensAt(walking(fra2gt_3),226). orientation(fra2gt_4,163,226). coords(fra2gt_4,197,209,226).  visible(fra2gt_4,226). happensAt(walking(fra2gt_4),226).
#show holdsAt/2.