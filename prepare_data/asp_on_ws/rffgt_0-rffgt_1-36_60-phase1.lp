
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

	orientation(rffgt_0,148,36). coords(rffgt_0,138,67,36).  visible(rffgt_0,36). happensAt(active(rffgt_0),36). orientation(rffgt_1,151,36). coords(rffgt_1,161,105,36).  visible(rffgt_1,36). happensAt(active(rffgt_1),36).
orientation(rffgt_0,144,37). coords(rffgt_0,138,67,37).  visible(rffgt_0,37). happensAt(active(rffgt_0),37). orientation(rffgt_1,151,37). coords(rffgt_1,162,106,37).  visible(rffgt_1,37). happensAt(active(rffgt_1),37).
orientation(rffgt_0,144,38). coords(rffgt_0,138,67,38).  visible(rffgt_0,38). happensAt(active(rffgt_0),38). orientation(rffgt_1,151,38). coords(rffgt_1,161,106,38).  visible(rffgt_1,38). happensAt(active(rffgt_1),38).
orientation(rffgt_0,144,39). coords(rffgt_0,138,66,39).  visible(rffgt_0,39). happensAt(active(rffgt_0),39). orientation(rffgt_1,151,39). coords(rffgt_1,161,106,39).  visible(rffgt_1,39). happensAt(active(rffgt_1),39).
orientation(rffgt_0,144,40). coords(rffgt_0,138,66,40).  visible(rffgt_0,40). happensAt(active(rffgt_0),40). orientation(rffgt_1,154,40). coords(rffgt_1,161,106,40).  visible(rffgt_1,40). happensAt(active(rffgt_1),40).
orientation(rffgt_0,141,41). coords(rffgt_0,138,66,41).  visible(rffgt_0,41). happensAt(active(rffgt_0),41). orientation(rffgt_1,156,41). coords(rffgt_1,161,106,41).  visible(rffgt_1,41). happensAt(active(rffgt_1),41).
orientation(rffgt_0,141,42). coords(rffgt_0,138,66,42).  visible(rffgt_0,42). happensAt(active(rffgt_0),42). orientation(rffgt_1,156,42). coords(rffgt_1,161,106,42).  visible(rffgt_1,42). happensAt(active(rffgt_1),42).
orientation(rffgt_0,141,43). coords(rffgt_0,138,65,43).  visible(rffgt_0,43). happensAt(active(rffgt_0),43). orientation(rffgt_1,156,43). coords(rffgt_1,161,106,43).  visible(rffgt_1,43). happensAt(active(rffgt_1),43).
orientation(rffgt_0,141,44). coords(rffgt_0,138,65,44).  visible(rffgt_0,44). happensAt(active(rffgt_0),44). orientation(rffgt_1,158,44). coords(rffgt_1,161,106,44).  visible(rffgt_1,44). happensAt(active(rffgt_1),44).
orientation(rffgt_0,137,45). coords(rffgt_0,138,65,45).  visible(rffgt_0,45). happensAt(active(rffgt_0),45). orientation(rffgt_1,160,45). coords(rffgt_1,161,106,45).  visible(rffgt_1,45). happensAt(active(rffgt_1),45).
orientation(rffgt_0,137,46). coords(rffgt_0,139,65,46).  visible(rffgt_0,46). happensAt(active(rffgt_0),46). orientation(rffgt_1,162,46). coords(rffgt_1,161,106,46).  visible(rffgt_1,46). happensAt(active(rffgt_1),46).
orientation(rffgt_0,137,47). coords(rffgt_0,139,65,47).  visible(rffgt_0,47). happensAt(active(rffgt_0),47). orientation(rffgt_1,167,47). coords(rffgt_1,161,106,47).  visible(rffgt_1,47). happensAt(active(rffgt_1),47).
orientation(rffgt_0,137,48). coords(rffgt_0,138,63,48).  visible(rffgt_0,48). happensAt(active(rffgt_0),48). orientation(rffgt_1,173,48). coords(rffgt_1,161,106,48).  visible(rffgt_1,48). happensAt(active(rffgt_1),48).
orientation(rffgt_0,137,49). coords(rffgt_0,138,63,49).  visible(rffgt_0,49). happensAt(active(rffgt_0),49). orientation(rffgt_1,173,49). coords(rffgt_1,161,106,49).  visible(rffgt_1,49). happensAt(active(rffgt_1),49).
orientation(rffgt_0,137,50). coords(rffgt_0,138,62,50).  visible(rffgt_0,50). happensAt(active(rffgt_0),50). orientation(rffgt_1,173,50). coords(rffgt_1,161,106,50).  visible(rffgt_1,50). happensAt(active(rffgt_1),50).
orientation(rffgt_0,137,51). coords(rffgt_0,138,62,51).  visible(rffgt_0,51). happensAt(active(rffgt_0),51). orientation(rffgt_1,171,51). coords(rffgt_1,161,106,51).  visible(rffgt_1,51). happensAt(active(rffgt_1),51).
orientation(rffgt_0,137,52). coords(rffgt_0,138,62,52).  visible(rffgt_0,52). happensAt(active(rffgt_0),52). orientation(rffgt_1,171,52). coords(rffgt_1,161,106,52).  visible(rffgt_1,52). happensAt(active(rffgt_1),52).
orientation(rffgt_0,137,53). coords(rffgt_0,136,61,53).  visible(rffgt_0,53). happensAt(active(rffgt_0),53). orientation(rffgt_1,167,53). coords(rffgt_1,161,106,53).  visible(rffgt_1,53). happensAt(active(rffgt_1),53).
orientation(rffgt_0,137,54). coords(rffgt_0,136,61,54).  visible(rffgt_0,54). happensAt(active(rffgt_0),54). orientation(rffgt_1,165,54). coords(rffgt_1,161,106,54).  visible(rffgt_1,54). happensAt(active(rffgt_1),54).
orientation(rffgt_0,137,55). coords(rffgt_0,136,61,55).  visible(rffgt_0,55). happensAt(active(rffgt_0),55). orientation(rffgt_1,161,55). coords(rffgt_1,160,105,55).  visible(rffgt_1,55). happensAt(active(rffgt_1),55).
orientation(rffgt_0,137,56). coords(rffgt_0,136,61,56).  visible(rffgt_0,56). happensAt(active(rffgt_0),56). orientation(rffgt_1,161,56). coords(rffgt_1,160,105,56).  visible(rffgt_1,56). happensAt(active(rffgt_1),56).
orientation(rffgt_0,137,57). coords(rffgt_0,136,61,57).  visible(rffgt_0,57). happensAt(active(rffgt_0),57). orientation(rffgt_1,158,57). coords(rffgt_1,160,105,57).  visible(rffgt_1,57). happensAt(active(rffgt_1),57).
orientation(rffgt_0,137,58). coords(rffgt_0,136,61,58).  visible(rffgt_0,58). happensAt(active(rffgt_0),58). orientation(rffgt_1,155,58). coords(rffgt_1,159,106,58).  visible(rffgt_1,58). happensAt(active(rffgt_1),58).
orientation(rffgt_0,137,59). coords(rffgt_0,136,61,59).  visible(rffgt_0,59). happensAt(active(rffgt_0),59). orientation(rffgt_1,155,59). coords(rffgt_1,158,107,59).  visible(rffgt_1,59). happensAt(active(rffgt_1),59).
orientation(rffgt_0,137,60). coords(rffgt_0,136,61,60).  visible(rffgt_0,60). happensAt(active(rffgt_0),60). orientation(rffgt_1,148,60). coords(rffgt_1,157,108,60).  visible(rffgt_1,60). happensAt(active(rffgt_1),60).
#show holdsAt/2.