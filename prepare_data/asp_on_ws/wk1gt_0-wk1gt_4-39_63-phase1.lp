
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

	orientation(wk1gt_0,14,39). coords(wk1gt_0,227,248,39).  visible(wk1gt_0,39). happensAt(walking(wk1gt_0),39). orientation(wk1gt_4,0,39). coords(wk1gt_4,23,212,39).  happensAt(appear(wk1gt_4),39). happensAt(walking(wk1gt_4),39).
orientation(wk1gt_0,13,40). coords(wk1gt_0,227,246,40).  visible(wk1gt_0,40). happensAt(walking(wk1gt_0),40). orientation(wk1gt_4,0,40). coords(wk1gt_4,23,212,40).  visible(wk1gt_4,40). happensAt(walking(wk1gt_4),40).
orientation(wk1gt_0,11,41). coords(wk1gt_0,227,243,41).  visible(wk1gt_0,41). happensAt(walking(wk1gt_0),41). orientation(wk1gt_4,0,41). coords(wk1gt_4,23,211,41).  visible(wk1gt_4,41). happensAt(walking(wk1gt_4),41).
orientation(wk1gt_0,11,42). coords(wk1gt_0,227,241,42).  visible(wk1gt_0,42). happensAt(walking(wk1gt_0),42). orientation(wk1gt_4,0,42). coords(wk1gt_4,23,211,42).  visible(wk1gt_4,42). happensAt(walking(wk1gt_4),42).
orientation(wk1gt_0,5,43). coords(wk1gt_0,227,240,43).  visible(wk1gt_0,43). happensAt(walking(wk1gt_0),43). orientation(wk1gt_4,0,43). coords(wk1gt_4,25,211,43).  visible(wk1gt_4,43). happensAt(walking(wk1gt_4),43).
orientation(wk1gt_0,5,44). coords(wk1gt_0,225,234,44).  visible(wk1gt_0,44). happensAt(walking(wk1gt_0),44). orientation(wk1gt_4,0,44). coords(wk1gt_4,25,211,44).  visible(wk1gt_4,44). happensAt(walking(wk1gt_4),44).
orientation(wk1gt_0,20,45). coords(wk1gt_0,224,230,45).  visible(wk1gt_0,45). happensAt(walking(wk1gt_0),45). orientation(wk1gt_4,0,45). coords(wk1gt_4,25,211,45).  visible(wk1gt_4,45). happensAt(walking(wk1gt_4),45).
orientation(wk1gt_0,18,46). coords(wk1gt_0,223,226,46).  visible(wk1gt_0,46). happensAt(walking(wk1gt_0),46). orientation(wk1gt_4,0,46). coords(wk1gt_4,26,210,46).  visible(wk1gt_4,46). happensAt(walking(wk1gt_4),46).
orientation(wk1gt_0,19,47). coords(wk1gt_0,223,223,47).  visible(wk1gt_0,47). happensAt(walking(wk1gt_0),47). orientation(wk1gt_4,0,47). coords(wk1gt_4,26,210,47).  visible(wk1gt_4,47). happensAt(walking(wk1gt_4),47).
orientation(wk1gt_0,19,48). coords(wk1gt_0,223,219,48).  visible(wk1gt_0,48). happensAt(walking(wk1gt_0),48). orientation(wk1gt_4,0,48). coords(wk1gt_4,26,210,48).  visible(wk1gt_4,48). happensAt(walking(wk1gt_4),48).
orientation(wk1gt_0,19,49). coords(wk1gt_0,223,218,49).  visible(wk1gt_0,49). happensAt(walking(wk1gt_0),49). orientation(wk1gt_4,0,49). coords(wk1gt_4,26,210,49).  visible(wk1gt_4,49). happensAt(walking(wk1gt_4),49).
orientation(wk1gt_0,6,50). coords(wk1gt_0,223,217,50).  visible(wk1gt_0,50). happensAt(walking(wk1gt_0),50). orientation(wk1gt_4,0,50). coords(wk1gt_4,27,209,50).  visible(wk1gt_4,50). happensAt(walking(wk1gt_4),50).
orientation(wk1gt_0,6,51). coords(wk1gt_0,223,216,51).  visible(wk1gt_0,51). happensAt(walking(wk1gt_0),51). orientation(wk1gt_4,0,51). coords(wk1gt_4,27,209,51).  visible(wk1gt_4,51). happensAt(walking(wk1gt_4),51).
orientation(wk1gt_0,5,52). coords(wk1gt_0,223,216,52).  visible(wk1gt_0,52). happensAt(walking(wk1gt_0),52). orientation(wk1gt_4,0,52). coords(wk1gt_4,27,208,52).  visible(wk1gt_4,52). happensAt(walking(wk1gt_4),52).
orientation(wk1gt_0,169,53). coords(wk1gt_0,223,216,53).  visible(wk1gt_0,53). happensAt(walking(wk1gt_0),53). orientation(wk1gt_4,0,53). coords(wk1gt_4,28,208,53).  visible(wk1gt_4,53). happensAt(walking(wk1gt_4),53).
orientation(wk1gt_0,158,54). coords(wk1gt_0,223,215,54).  visible(wk1gt_0,54). happensAt(walking(wk1gt_0),54). orientation(wk1gt_4,0,54). coords(wk1gt_4,28,208,54).  visible(wk1gt_4,54). happensAt(walking(wk1gt_4),54).
orientation(wk1gt_0,155,55). coords(wk1gt_0,223,214,55).  visible(wk1gt_0,55). happensAt(walking(wk1gt_0),55). orientation(wk1gt_4,0,55). coords(wk1gt_4,28,207,55).  visible(wk1gt_4,55). happensAt(walking(wk1gt_4),55).
orientation(wk1gt_0,158,56). coords(wk1gt_0,223,209,56).  visible(wk1gt_0,56). happensAt(walking(wk1gt_0),56). orientation(wk1gt_4,0,56). coords(wk1gt_4,29,207,56).  visible(wk1gt_4,56). happensAt(walking(wk1gt_4),56).
orientation(wk1gt_0,155,57). coords(wk1gt_0,223,204,57).  visible(wk1gt_0,57). happensAt(walking(wk1gt_0),57). orientation(wk1gt_4,0,57). coords(wk1gt_4,30,207,57).  visible(wk1gt_4,57). happensAt(walking(wk1gt_4),57).
orientation(wk1gt_0,155,58). coords(wk1gt_0,223,203,58).  visible(wk1gt_0,58). happensAt(walking(wk1gt_0),58). orientation(wk1gt_4,0,58). coords(wk1gt_4,31,207,58).  visible(wk1gt_4,58). happensAt(walking(wk1gt_4),58).
orientation(wk1gt_0,155,59). coords(wk1gt_0,223,201,59).  visible(wk1gt_0,59). happensAt(walking(wk1gt_0),59). orientation(wk1gt_4,0,59). coords(wk1gt_4,31,206,59).  visible(wk1gt_4,59). happensAt(walking(wk1gt_4),59).
orientation(wk1gt_0,155,60). coords(wk1gt_0,223,199,60).  visible(wk1gt_0,60). happensAt(walking(wk1gt_0),60). orientation(wk1gt_4,0,60). coords(wk1gt_4,31,206,60).  visible(wk1gt_4,60). happensAt(walking(wk1gt_4),60).
orientation(wk1gt_0,148,61). coords(wk1gt_0,223,197,61).  visible(wk1gt_0,61). happensAt(walking(wk1gt_0),61). orientation(wk1gt_4,0,61). coords(wk1gt_4,35,205,61).  visible(wk1gt_4,61). happensAt(walking(wk1gt_4),61).
orientation(wk1gt_0,148,62). coords(wk1gt_0,223,194,62).  visible(wk1gt_0,62). happensAt(walking(wk1gt_0),62). orientation(wk1gt_4,0,62). coords(wk1gt_4,35,204,62).  visible(wk1gt_4,62). happensAt(walking(wk1gt_4),62).
orientation(wk1gt_0,143,63). coords(wk1gt_0,223,192,63).  visible(wk1gt_0,63). happensAt(walking(wk1gt_0),63). orientation(wk1gt_4,0,63). coords(wk1gt_4,35,204,63).  visible(wk1gt_4,63). happensAt(walking(wk1gt_4),63).
#show holdsAt/2.