
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

	orientation(lb2gt_1,109,36). coords(lb2gt_1,240,28,36).  visible(lb2gt_1,36). happensAt(walking(lb2gt_1),36). orientation(lb2gt_2,147,36). coords(lb2gt_2,20,53,36).  visible(lb2gt_2,36). happensAt(walking(lb2gt_2),36).
orientation(lb2gt_1,107,37). coords(lb2gt_1,241,27,37).  visible(lb2gt_1,37). happensAt(walking(lb2gt_1),37). orientation(lb2gt_2,147,37). coords(lb2gt_2,20,53,37).  visible(lb2gt_2,37). happensAt(walking(lb2gt_2),37).
orientation(lb2gt_1,108,38). coords(lb2gt_1,242,25,38).  visible(lb2gt_1,38). happensAt(walking(lb2gt_1),38). orientation(lb2gt_2,147,38). coords(lb2gt_2,20,54,38).  visible(lb2gt_2,38). happensAt(walking(lb2gt_2),38).
orientation(lb2gt_1,111,39). coords(lb2gt_1,242,24,39).  visible(lb2gt_1,39). happensAt(walking(lb2gt_1),39). orientation(lb2gt_2,147,39). coords(lb2gt_2,20,53,39).  visible(lb2gt_2,39). happensAt(walking(lb2gt_2),39).
orientation(lb2gt_1,107,40). coords(lb2gt_1,242,22,40).  visible(lb2gt_1,40). happensAt(walking(lb2gt_1),40). orientation(lb2gt_2,147,40). coords(lb2gt_2,20,53,40).  visible(lb2gt_2,40). happensAt(walking(lb2gt_2),40).
orientation(lb2gt_1,112,41). coords(lb2gt_1,243,20,41).  visible(lb2gt_1,41). happensAt(walking(lb2gt_1),41). orientation(lb2gt_2,147,41). coords(lb2gt_2,20,53,41).  visible(lb2gt_2,41). happensAt(walking(lb2gt_2),41).
orientation(lb2gt_1,116,42). coords(lb2gt_1,243,19,42).  visible(lb2gt_1,42). happensAt(walking(lb2gt_1),42). orientation(lb2gt_2,147,42). coords(lb2gt_2,20,54,42).  visible(lb2gt_2,42). happensAt(walking(lb2gt_2),42).
orientation(lb2gt_1,116,43). coords(lb2gt_1,245,18,43).  visible(lb2gt_1,43). happensAt(walking(lb2gt_1),43). orientation(lb2gt_2,147,43). coords(lb2gt_2,19,54,43).  visible(lb2gt_2,43). happensAt(walking(lb2gt_2),43).
orientation(lb2gt_1,113,44). coords(lb2gt_1,246,18,44).  visible(lb2gt_1,44). happensAt(walking(lb2gt_1),44). orientation(lb2gt_2,147,44). coords(lb2gt_2,19,54,44).  visible(lb2gt_2,44). happensAt(walking(lb2gt_2),44).
orientation(lb2gt_1,114,45). coords(lb2gt_1,246,18,45).  visible(lb2gt_1,45). happensAt(walking(lb2gt_1),45). orientation(lb2gt_2,147,45). coords(lb2gt_2,19,54,45).  visible(lb2gt_2,45). happensAt(walking(lb2gt_2),45).
orientation(lb2gt_1,117,46). coords(lb2gt_1,247,18,46).  visible(lb2gt_1,46). happensAt(walking(lb2gt_1),46). orientation(lb2gt_2,147,46). coords(lb2gt_2,19,54,46).  visible(lb2gt_2,46). happensAt(walking(lb2gt_2),46).
orientation(lb2gt_1,108,47). coords(lb2gt_1,248,18,47).  visible(lb2gt_1,47). happensAt(walking(lb2gt_1),47). orientation(lb2gt_2,147,47). coords(lb2gt_2,19,54,47).  visible(lb2gt_2,47). happensAt(walking(lb2gt_2),47).
orientation(lb2gt_1,110,48). coords(lb2gt_1,249,18,48).  visible(lb2gt_1,48). happensAt(walking(lb2gt_1),48). orientation(lb2gt_2,147,48). coords(lb2gt_2,19,54,48).  visible(lb2gt_2,48). happensAt(walking(lb2gt_2),48).
orientation(lb2gt_1,104,49). coords(lb2gt_1,250,18,49).  visible(lb2gt_1,49). happensAt(walking(lb2gt_1),49). orientation(lb2gt_2,147,49). coords(lb2gt_2,19,54,49).  visible(lb2gt_2,49). happensAt(walking(lb2gt_2),49).
orientation(lb2gt_1,104,50). coords(lb2gt_1,251,18,50).  visible(lb2gt_1,50). happensAt(walking(lb2gt_1),50). orientation(lb2gt_2,147,50). coords(lb2gt_2,19,54,50).  visible(lb2gt_2,50). happensAt(walking(lb2gt_2),50).
orientation(lb2gt_1,104,51). coords(lb2gt_1,252,17,51).  visible(lb2gt_1,51). happensAt(walking(lb2gt_1),51). orientation(lb2gt_2,147,51). coords(lb2gt_2,19,54,51).  visible(lb2gt_2,51). happensAt(walking(lb2gt_2),51).
orientation(lb2gt_1,104,52). coords(lb2gt_1,252,16,52).  visible(lb2gt_1,52). happensAt(walking(lb2gt_1),52). orientation(lb2gt_2,147,52). coords(lb2gt_2,19,54,52).  visible(lb2gt_2,52). happensAt(walking(lb2gt_2),52).
orientation(lb2gt_1,104,53). coords(lb2gt_1,253,15,53).  visible(lb2gt_1,53). happensAt(walking(lb2gt_1),53). orientation(lb2gt_2,147,53). coords(lb2gt_2,19,54,53).  visible(lb2gt_2,53). happensAt(walking(lb2gt_2),53).
orientation(lb2gt_1,104,54). coords(lb2gt_1,255,13,54).  visible(lb2gt_1,54). happensAt(walking(lb2gt_1),54). orientation(lb2gt_2,147,54). coords(lb2gt_2,19,54,54).  visible(lb2gt_2,54). happensAt(walking(lb2gt_2),54).
orientation(lb2gt_1,104,55). coords(lb2gt_1,255,12,55).  visible(lb2gt_1,55). happensAt(walking(lb2gt_1),55). orientation(lb2gt_2,147,55). coords(lb2gt_2,19,54,55).  visible(lb2gt_2,55). happensAt(walking(lb2gt_2),55).
orientation(lb2gt_1,104,56). coords(lb2gt_1,256,10,56).  visible(lb2gt_1,56). happensAt(walking(lb2gt_1),56). orientation(lb2gt_2,147,56). coords(lb2gt_2,19,54,56).  visible(lb2gt_2,56). happensAt(walking(lb2gt_2),56).
orientation(lb2gt_1,104,57). coords(lb2gt_1,256,10,57).  visible(lb2gt_1,57). happensAt(walking(lb2gt_1),57). orientation(lb2gt_2,147,57). coords(lb2gt_2,19,54,57).  visible(lb2gt_2,57). happensAt(walking(lb2gt_2),57).
orientation(lb2gt_1,104,58). coords(lb2gt_1,256,10,58).  visible(lb2gt_1,58). happensAt(walking(lb2gt_1),58). orientation(lb2gt_2,147,58). coords(lb2gt_2,19,54,58).  visible(lb2gt_2,58). happensAt(walking(lb2gt_2),58).
orientation(lb2gt_1,104,59). coords(lb2gt_1,256,10,59).  visible(lb2gt_1,59). happensAt(walking(lb2gt_1),59). orientation(lb2gt_2,147,59). coords(lb2gt_2,19,54,59).  visible(lb2gt_2,59). happensAt(walking(lb2gt_2),59).
orientation(lb2gt_1,104,60). coords(lb2gt_1,257,10,60).  visible(lb2gt_1,60). happensAt(walking(lb2gt_1),60). orientation(lb2gt_2,147,60). coords(lb2gt_2,19,54,60).  happensAt(disappear(lb2gt_2),60). happensAt(walking(lb2gt_2),60).
#show holdsAt/2.